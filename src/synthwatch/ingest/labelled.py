"""Loaders for the public labelled datasets the ensemble will train on.

Two sources, with very different shapes and very different meanings.

**Indiana Bot Repository** ships annotation files: a list of account ids and a
class, with the actual posts distributed separately or not at all. The labels
come from several studies with different annotation procedures, and the class
vocabulary varies between them -- ``bot``/``human`` in one file, ``0``/``1`` in
another, ``social_spambot_1`` in a third.

**The Twitter Information Operations Archive** ships the data itself: every
tweet and profile from a set of accounts a platform removed and attributed to a
state-linked operation. There is no negative class in it at all.

That asymmetry is the most important thing in this module, so it is enforced
rather than mentioned: :func:`label_coverage` refuses to stay quiet about a
single-class label set, because a model trained on takedown data plus a control
group collected some other way learns to tell the two *collections* apart, and
reports a beautiful AUC for doing it.

Numeric classes are never guessed. ``1`` means bot in Varol's file and the
opposite convention exists elsewhere; a loader that picks one silently inverts
an entire training set, and nothing downstream would look wrong.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, tzinfo
from pathlib import Path
from typing import Any, Final, Literal

from synthwatch.ingest.base import IngestReport, LoadResult
from synthwatch.ingest.native import NativeAdapter
from synthwatch.models import Corpus, LabelRecord
from synthwatch.types import AccountId, Label, LabelMethod, Platform, PostKind

__all__ = [
    "BOT_REPOSITORY_CLASSES",
    "IO_ARCHIVE_POST_ALIASES",
    "IOArchiveAdapter",
    "LabelCoverage",
    "NumericConvention",
    "UnknownClassError",
    "attach_labels",
    "label_coverage",
    "read_label_table",
]

NumericConvention = Literal["1_is_bot", "0_is_bot"]
"""Which way round a numeric annotation file runs. There is no default."""

BOT_REPOSITORY_CLASSES: Final[Mapping[str, Label]] = {
    "bot": Label.AUTOMATED,
    "bots": Label.AUTOMATED,
    "spambot": Label.AUTOMATED,
    "social_spambot": Label.AUTOMATED,
    "traditional_spambot": Label.AUTOMATED,
    "fake_follower": Label.AUTOMATED,
    "fakefollower": Label.AUTOMATED,
    "self_declared": Label.AUTOMATED,
    "human": Label.ORGANIC,
    "humans": Label.ORGANIC,
    "genuine": Label.ORGANIC,
    "genuine_account": Label.ORGANIC,
    "real": Label.ORGANIC,
    "verified": Label.ORGANIC,
}
"""Class strings seen across Bot Repository files, mapped onto the schema.

Suffixed variants (``social_spambot_1``, ``fake_followers_2``) are matched by
prefix. Anything unrecognised raises :class:`UnknownClassError` rather than
falling through to ``UNKNOWN``: a silently discarded class is a silently
shrunken training set.
"""

IO_ARCHIVE_POST_ALIASES: Final[Mapping[str, str]] = {
    "tweetid": "post_id",
    "userid": "account_id",
    "tweet_time": "created_at",
    "tweet_text": "text",
    "tweet_language": "language",
    "in_reply_to_tweetid": "parent_post_id",
    "tweet_client_name": "client",
    "user_mentions": "mentions",
    "quote_count": "reply_count",
}

IO_ARCHIVE_ACCOUNT_ALIASES: Final[Mapping[str, str]] = {
    "userid": "account_id",
    "user_screen_name": "handle",
    "user_display_name": "display_name",
    "user_profile_description": "description",
    "user_reported_location": "location",
    "account_creation_date": "created_at",
    "account_language": "language",
    "follower_count": "followers_count",
    "following_count": "following_count",
}

MIN_CLASSES_FOR_SUPERVISION: Final = 2
"""Below this, a label set describes a collection rather than a behaviour."""


class UnknownClassError(ValueError):
    """Raised when an annotation file uses a class the loader does not know."""


def _normalise_class(raw: str, convention: NumericConvention | None) -> Label:
    """Map one raw class string onto a :class:`Label`.

    Raises:
        UnknownClassError: If the class is unrecognised, or numeric with no
            declared convention.
    """
    text = raw.strip().casefold().replace("-", "_").replace(" ", "_")
    if not text:
        msg = "empty class value"
        raise UnknownClassError(msg)

    if text in {"0", "1"}:
        if convention is None:
            msg = (
                f"numeric class {raw!r} with no declared convention. Pass "
                "numeric_convention='1_is_bot' or '0_is_bot': the two exist in "
                "the wild, and picking one silently inverts the training set."
            )
            raise UnknownClassError(msg)
        positive = "1" if convention == "1_is_bot" else "0"
        return Label.AUTOMATED if text == positive else Label.ORGANIC

    if text in BOT_REPOSITORY_CLASSES:
        return BOT_REPOSITORY_CLASSES[text]
    for prefix, label in BOT_REPOSITORY_CLASSES.items():
        if text.startswith(f"{prefix}_"):
            return label

    msg = (
        f"unrecognised class {raw!r}. Extend BOT_REPOSITORY_CLASSES rather than "
        "letting it fall through: a class quietly dropped is a training set "
        "quietly shrunk."
    )
    raise UnknownClassError(msg)


def read_label_table(
    path: Path,
    *,
    dataset: str,
    method: LabelMethod = LabelMethod.HUMAN_ANNOTATION,
    platform: Platform = Platform.TWITTER,
    numeric_convention: NumericConvention | None = None,
    delimiter: str | None = None,
    id_column: str | None = None,
    class_column: str | None = None,
) -> tuple[LabelRecord, ...]:
    """Read an annotation file into :class:`LabelRecord` objects.

    Handles both shapes the Bot Repository ships: a headerless two-column file
    of ``id<TAB>class``, and a CSV with named columns.

    Args:
        path: The annotation file.
        dataset: Provenance string recorded on every label, e.g.
            ``"indiana-bot-repository/varol-2017"``. Required, because an
            evaluation that cannot say which annotation procedure produced its
            labels is not reproducible.
        method: How the labels were produced.
        platform: Platform the account ids belong to.
        numeric_convention: Required when the file uses ``0``/``1``.
        delimiter: Field separator; inferred from the suffix when omitted.
        id_column: Name of the id column in a file with a header.
        class_column: Name of the class column in a file with a header.

    Raises:
        UnknownClassError: On an unrecognised or ambiguous class value.
        ValueError: If the file has fewer than two usable columns.
    """
    separator = delimiter or ("," if path.suffix.casefold() == ".csv" else "\t")
    records: list[LabelRecord] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=separator))

    if not rows:
        return ()

    start = 0
    id_index, class_index = 0, 1
    header = [cell.strip().casefold() for cell in rows[0]]
    if id_column or class_column or _looks_like_header(header):
        wanted_id = (id_column or "user_id").casefold()
        wanted_class = (class_column or "label").casefold()
        id_index = _column_index(header, wanted_id, default=0)
        class_index = _column_index(header, wanted_class, default=1)
        start = 1

    for number, row in enumerate(rows[start:], start=start + 1):
        if len(row) <= max(id_index, class_index):
            if not any(cell.strip() for cell in row):
                continue
            msg = f"{path.name}:{number}: expected at least two columns, got {len(row)}"
            raise ValueError(msg)
        account_id = row[id_index].strip()
        if not account_id:
            continue
        records.append(
            LabelRecord(
                account_id=account_id,
                platform=platform,
                label=_normalise_class(row[class_index], numeric_convention),
                dataset=dataset,
                method=method,
                notes=f"source class: {row[class_index].strip()}",
            )
        )
    return tuple(records)


HEADER_NAMES: Final = frozenset({"user_id", "userid", "id", "label", "class", "account_id"})


def _looks_like_header(cells: Sequence[str]) -> bool:
    """Whether the first row names columns rather than holding data."""
    return any(cell in HEADER_NAMES for cell in cells)


def _column_index(header: Sequence[str], wanted: str, *, default: int) -> int:
    """Index of ``wanted`` in ``header``, falling back to a position."""
    return header.index(wanted) if wanted in header else default


# --------------------------------------------------------------------------
# Twitter Information Operations Archive
# --------------------------------------------------------------------------


def _io_archive_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the fields the archive encodes as flags rather than columns."""
    derived = dict(row)
    is_retweet = str(row.get("is_retweet", "")).strip().casefold() in {"true", "1", "yes"}
    replied_to = str(row.get("in_reply_to_tweetid", "")).strip()
    quoted = str(row.get("quoted_tweet_tweetid", "")).strip()
    if is_retweet:
        derived["kind"] = PostKind.REPOST.value
    elif replied_to:
        derived["kind"] = PostKind.REPLY.value
    elif quoted:
        derived["kind"] = PostKind.QUOTE.value
    else:
        derived["kind"] = PostKind.ORIGINAL.value
    return derived


class IOArchiveAdapter:
    """Reads a Twitter Information Operations Archive takedown.

    Every account in a takedown was removed by the platform and attributed to
    an influence operation, so the loader attaches a
    :class:`~synthwatch.models.LabelRecord` to each of them with
    ``method=PLATFORM_ENFORCEMENT``. That method matters: the label reflects a
    platform's enforcement decision and its policy, not an independent
    determination, and it carries that platform's error rate with it.

    ``INFO_OPERATION`` is deliberately not a synonym for ``AUTOMATED``. Many
    accounts in these takedowns were operated by people, full time, by hand.
    """

    name = "twitter-io-archive"
    platform = Platform.TWITTER

    def __init__(self, *, dataset: str, timezone: tzinfo = UTC, strict: bool = False) -> None:
        """Build the adapter.

        Args:
            dataset: Provenance for the labels, e.g.
                ``"twitter-io-archive/2019-06-iran"``. Required: takedowns
                differ enormously between countries and dates, and pooling them
                without saying so is not reproducible.
            timezone: What the archive's offset-free timestamps mean. Defaults
                to UTC because that is what the archive's own documentation
                says they are. This is a *declaration*, made once, in the
                adapter that knows the source -- which is the difference
                between stating a fact about a format and having a parser guess
                per row. Override it if a particular release says otherwise.
            strict: Raise on the first unparsable row instead of counting it.
        """
        self.dataset = dataset
        self._adapter = NativeAdapter(
            platform=Platform.TWITTER,
            assume_timezone=timezone,
            post_aliases=IO_ARCHIVE_POST_ALIASES,
            account_aliases=IO_ARCHIVE_ACCOUNT_ALIASES,
            post_row_transform=_io_archive_row,
            strict=strict,
        )

    def sniff(self, path: Path) -> bool:
        """Whether the filename matches the archive's naming convention."""
        name = path.name.casefold()
        return "csv_hashed" in name or ("tweets" in name and name.endswith(".csv"))

    def load(self, path: Path) -> LoadResult:
        """Read a takedown's tweets file, labelling every account in it."""
        return self.load_takedown(path)

    def load_takedown(self, tweets: Path, users: Path | None = None) -> LoadResult:
        """Read a takedown's tweets and, when available, its profiles.

        Returns:
            A load result whose corpus already carries one label per account.
        """
        loaded = self._adapter.load_tables(tweets, users)
        corpus = loaded.corpus
        account_ids = sorted(
            {account.account_id for account in corpus.accounts} | set(corpus.posts_by_account)
        )
        labels = tuple(
            LabelRecord(
                account_id=account_id,
                platform=Platform.TWITTER,
                label=Label.INFO_OPERATION,
                dataset=self.dataset,
                method=LabelMethod.PLATFORM_ENFORCEMENT,
                notes="present in a platform takedown; not a synonym for automated",
            )
            for account_id in account_ids
        )
        labelled = Corpus(
            accounts=corpus.accounts,
            posts=corpus.posts,
            labels=labels,
            source=f"{self.dataset} ({loaded.report.source})",
            collected_at=corpus.collected_at,
        )
        report = IngestReport(
            source=loaded.report.source,
            n_accounts=loaded.report.n_accounts,
            n_posts=loaded.report.n_posts,
            n_skipped=loaded.report.n_skipped,
            skip_reasons=loaded.report.skip_reasons,
            warnings=(
                *loaded.report.warnings,
                f"every account here carries one class ({Label.INFO_OPERATION.value}); "
                "a control group must come from elsewhere, and that difference in "
                "collection is a confound, not a baseline",
            ),
        )
        return LoadResult(corpus=labelled, report=report)


# --------------------------------------------------------------------------
# Joining labels to a corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelCoverage:
    """How much of a corpus a label set actually reaches.

    Attributes:
        n_labels: Labels supplied.
        n_matched: Labels whose account appears in the corpus.
        n_unmatched: Labels for accounts the corpus does not contain.
        n_unlabelled_accounts: Corpus accounts with no label.
        counts: Accounts per class, among the matched labels.
        datasets: Provenance strings present in the matched labels.
        warnings: Conditions that would make a trained model untrustworthy.
    """

    n_labels: int
    n_matched: int
    n_unmatched: int
    n_unlabelled_accounts: int
    counts: Mapping[Label, int]
    datasets: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def match_rate(self) -> float:
        """Share of supplied labels that reached an account in the corpus."""
        return self.n_matched / self.n_labels if self.n_labels else 0.0

    @property
    def minority_share(self) -> float:
        """Share held by the smallest class, or ``0`` below two classes."""
        total = sum(self.counts.values())
        if total == 0 or len(self.counts) < MIN_CLASSES_FOR_SUPERVISION:
            return 0.0
        return min(self.counts.values()) / total

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the report layer."""
        return {
            "n_labels": self.n_labels,
            "n_matched": self.n_matched,
            "n_unmatched": self.n_unmatched,
            "n_unlabelled_accounts": self.n_unlabelled_accounts,
            "match_rate": round(self.match_rate, 4),
            "counts": {label.value: count for label, count in self.counts.items()},
            "minority_share": round(self.minority_share, 4),
            "datasets": list(self.datasets),
            "warnings": list(self.warnings),
        }


LOW_MATCH_RATE: Final = 0.5
"""Below this, the labels and the corpus are mostly describing different accounts."""

SEVERE_IMBALANCE: Final = 0.05
"""Below this minority share, accuracy is meaningless and so is an untuned model."""


def label_coverage(corpus: Corpus, labels: Sequence[LabelRecord] | None = None) -> LabelCoverage:
    """Measure how well a label set covers a corpus, and what could go wrong.

    The warnings are the point. Every one of them describes a situation where a
    model will train happily and report a good score while having learned
    something other than the thing it was supposed to learn.
    """
    records = tuple(labels if labels is not None else corpus.labels)
    known = set(corpus.accounts_by_id) | set(corpus.posts_by_account)
    matched = [record for record in records if record.account_id in known]
    counts = Counter(record.label for record in matched)
    labelled_accounts = {record.account_id for record in matched}

    warnings: list[str] = []
    if len(counts) < MIN_CLASSES_FOR_SUPERVISION:
        present = ", ".join(sorted(label.value for label in counts)) or "none"
        warnings.append(
            f"single-class label set ({present}). A classifier trained on this plus a "
            "control group gathered another way learns to tell the two collections "
            "apart, and will report an excellent score for doing it."
        )
    elif 0 < min(counts.values()) / sum(counts.values()) < SEVERE_IMBALANCE:
        warnings.append(
            "severe class imbalance: report precision and recall per class, and "
            "calibrate; accuracy here is a measure of the majority class."
        )
    if records and len(matched) / len(records) < LOW_MATCH_RATE:
        warnings.append(
            f"only {len(matched)} of {len(records)} labels matched an account in this "
            "corpus. Annotation files reference accounts that have since been "
            "deleted, and the survivors are not a random sample of the original set."
        )
    datasets = sorted({record.dataset for record in matched})
    if len(datasets) > 1:
        warnings.append(
            f"labels come from {len(datasets)} datasets with different annotation "
            "procedures. Pooling them mixes their error modes; report per-dataset "
            "performance as well."
        )

    return LabelCoverage(
        n_labels=len(records),
        n_matched=len(matched),
        n_unmatched=len(records) - len(matched),
        n_unlabelled_accounts=len(known - labelled_accounts),
        counts=dict(counts),
        datasets=tuple(datasets),
        warnings=tuple(warnings),
    )


def attach_labels(
    corpus: Corpus, labels: Iterable[LabelRecord], *, replace: bool = False
) -> tuple[Corpus, LabelCoverage]:
    """Attach labels to a corpus and report how far they reach.

    Args:
        corpus: The corpus to label.
        labels: Labels to attach. Only those matching an account in the corpus
            are kept; the rest are counted in the coverage report.
        replace: Discard any labels the corpus already carries.

    Returns:
        The labelled corpus and its coverage report.
    """
    incoming = tuple(labels)
    known = set(corpus.accounts_by_id) | set(corpus.posts_by_account)
    existing: tuple[LabelRecord, ...] = () if replace else corpus.labels
    considered = (*existing, *incoming)
    merged: dict[tuple[AccountId, str], LabelRecord] = {
        (record.account_id, record.dataset): record
        for record in considered
        if record.account_id in known
    }
    labelled = Corpus(
        accounts=corpus.accounts,
        posts=corpus.posts,
        labels=tuple(merged.values()),
        source=corpus.source,
        collected_at=corpus.collected_at,
        notes=corpus.notes,
    )
    # Coverage is measured against everything that was offered, not against
    # what survived the join. Reading it off the attached labels would report a
    # perfect match rate every time and hide the accounts that went missing.
    return labelled, label_coverage(labelled, considered)
