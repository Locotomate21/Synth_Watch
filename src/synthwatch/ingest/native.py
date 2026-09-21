"""The native adapter: CSV, JSON and JSONL in SynthWatch's own schema.

This is the adapter every other one is measured against, and the only entry
point that does not need a platform API. Point it at an export you already
have and it produces a :class:`~synthwatch.models.Corpus`.

Two properties matter more than convenience here.

**Nothing is dropped silently.** A row that cannot be parsed is counted, with a
reason, in the :class:`~synthwatch.ingest.base.IngestReport` that travels with
the corpus. A 30% drop rate is a finding about the data, not an implementation
detail to hide behind a ``try``. Columns the schema has no field for are kept
verbatim in ``extra``.

**Timezones are never guessed.** A timestamp without an offset is refused
unless the caller states, explicitly, which timezone the file is in. Assuming
UTC would silently rotate every circadian and burst feature downstream, and
the resulting numbers would look perfectly reasonable.

Accepted layouts
----------------

* ``corpus.json`` -- an object with ``accounts``, ``posts`` and optional
  ``labels`` arrays. This is what :func:`write_corpus` produces, so a
  normalised corpus can be cached and re-read without re-parsing the source.
* ``posts.jsonl`` -- one post object per line.
* ``posts.csv`` / ``accounts.csv`` -- a table each, loaded together by
  :meth:`NativeAdapter.load_tables`.
* A bare JSON array is read as posts.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from pydantic import ValidationError

from synthwatch.ingest.base import REGISTRY, IngestReport, LoadResult
from synthwatch.models import Account, Corpus, LabelRecord, Post
from synthwatch.types import Platform

__all__ = [
    "DEFAULT_ACCOUNT_ALIASES",
    "DEFAULT_POST_ALIASES",
    "AmbiguousDateError",
    "CorruptedIdentifierError",
    "DateOrder",
    "NaiveTimestampError",
    "NativeAdapter",
    "parse_timestamp",
    "timezone_of",
    "write_corpus",
]

DateOrder = Literal["dmy", "mdy"]
"""Which way round a file writes slash dates. There is no default."""

_Record = TypeVar("_Record")

JSON_SUFFIXES: Final = frozenset({".json"})
JSONL_SUFFIXES: Final = frozenset({".jsonl", ".ndjson"})
TABLE_SUFFIXES: Final = frozenset({".csv", ".tsv"})

MILLISECOND_EPOCH_CUTOFF: Final = 1e11
"""Epoch values above this are milliseconds: 1e11 seconds is the year 5138."""

HIGH_DROP_RATE: Final = 0.1
"""Above this share of skipped rows, the load itself is worth reporting."""

OVERFLOW_KEY: Final = "__overflow__"
"""Where csv puts fields a row has beyond its header.

Naming it matters: left at the default, those land under a ``None`` key and
every later lookup breaks on a key that is not a string. A row with more
fields than its header is ragged -- usually an unquoted delimiter inside a
value -- and half-parsing it would be worse than counting it as skipped.
"""

DEFAULT_POST_ALIASES: Final[Mapping[str, str]] = {
    "id": "post_id",
    "status_id": "post_id",
    "tweet_id": "post_id",
    "tweetid": "post_id",
    "author": "account_id",
    "author_id": "account_id",
    "user": "account_id",
    "user_id": "account_id",
    "userid": "account_id",
    "created": "created_at",
    "date": "created_at",
    "time": "created_at",
    "timestamp": "created_at",
    "tweet_time": "created_at",
    "body": "text",
    "content": "text",
    "full_text": "text",
    "message": "text",
    "tweet_text": "text",
    "lang": "language",
    "in_reply_to": "parent_post_id",
    "parent_id": "parent_post_id",
    "conversation_id": "root_post_id",
    "thread_id": "root_post_id",
    "favorite_count": "like_count",
    "likes": "like_count",
    "retweet_count": "repost_count",
    "shares": "repost_count",
    "replies": "reply_count",
    "source": "client",
}
"""Column names seen in the wild, mapped onto schema fields.

Drawn from the Twitter Information Operations Archive, Pushshift-style Reddit
dumps and the usual spreadsheet exports. Ambiguous names are deliberately
absent: ``name`` could be a handle or a display name, so it is left alone
rather than guessed at.
"""

DEFAULT_ACCOUNT_ALIASES: Final[Mapping[str, str]] = {
    "id": "account_id",
    "user_id": "account_id",
    "userid": "account_id",
    "author_id": "account_id",
    "acct": "handle",
    "screen_name": "handle",
    "username": "handle",
    "user_screen_name": "handle",
    "display": "display_name",
    "user_display_name": "display_name",
    "account_created_at": "created_at",
    "created": "created_at",
    "joined": "created_at",
    "bio": "description",
    "profile_description": "description",
    "user_profile_description": "description",
    "followers": "followers_count",
    "follower_count": "followers_count",
    "friends_count": "following_count",
    "following": "following_count",
    "posts": "post_count",
    "statuses_count": "post_count",
    "tweet_count": "post_count",
}

_BOOL_TRUE: Final = frozenset({"true", "t", "yes", "y", "1"})
_BOOL_FALSE: Final = frozenset({"false", "f", "no", "n", "0"})

_POST_INT_FIELDS: Final = ("media_count", "like_count", "repost_count", "reply_count")
_ACCOUNT_INT_FIELDS: Final = ("followers_count", "following_count", "post_count")
_ACCOUNT_BOOL_FIELDS: Final = (
    "verified",
    "protected",
    "has_default_profile_image",
    "declared_automated",
)
_POST_LIST_FIELDS: Final = ("urls", "hashtags", "mentions")


class NaiveTimestampError(ValueError):
    """Raised when a timestamp has no offset and none was declared."""


class AmbiguousDateError(ValueError):
    """Raised when a slash date could be read two ways and none was declared."""


class CorruptedIdentifierError(ValueError):
    """Raised when an identifier has visibly lost precision.

    A spreadsheet that opens a CSV of 18-digit account ids and saves it again
    rewrites them as ``1.01421E+18``. The value looks like data, joins to
    nothing, and silently drops that account out of every merge it takes part
    in. Real published datasets contain rows in exactly this state.
    """


def parse_timestamp(
    value: object,
    *,
    assume_timezone: tzinfo | None = None,
    date_order: DateOrder | None = None,
) -> datetime:
    """Parse a timestamp from a CSV cell or a JSON value.

    Accepts ISO 8601 strings (including a trailing ``Z``), epoch seconds and
    epoch milliseconds, and ``datetime`` objects passed straight through.

    Args:
        value: The raw value.
        assume_timezone: Timezone to attach to a naive timestamp. Leave it
            ``None`` to refuse naive input, which is the default because a
            wrong offset is invisible downstream and corrupts every temporal
            feature.
        date_order: How to read a slash date whose first two components are
            both 12 or below. Leave it ``None`` to refuse those.

    Raises:
        NaiveTimestampError: If the value has no offset and none was declared.
        AmbiguousDateError: If a slash date could be read two ways.
        ValueError: If the value cannot be parsed at all.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if abs(seconds) > MILLISECOND_EPOCH_CUTOFF:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC)
    else:
        raw = str(value).strip()
        if not raw:
            msg = "empty timestamp"
            raise ValueError(msg)
        if raw.lstrip("-").replace(".", "", 1).isdigit():
            return parse_timestamp(
                float(raw), assume_timezone=assume_timezone, date_order=date_order
            )
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            parsed = _parse_common_formats(raw, date_order)

    if parsed.tzinfo is None:
        if assume_timezone is None:
            msg = (
                f"timestamp {value!r} has no UTC offset. Pass assume_timezone to state "
                "which timezone this file is in; guessing would silently rotate every "
                "circadian and inter-arrival feature."
            )
            raise NaiveTimestampError(msg)
        parsed = parsed.replace(tzinfo=assume_timezone)
    return parsed


UNAMBIGUOUS_MONTH = 12
"""A slash-date component above this can only be a day."""

_SLASH_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")

_ISO_FORMATS: Final = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%a %b %d %H:%M:%S %z %Y",  # legacy Twitter
)


def _parse_slash_date(match: re.Match[str], date_order: DateOrder | None) -> datetime:
    """Parse ``7/3/2018``, refusing when the order is genuinely ambiguous.

    ``13/7/2018`` can only be day-first and ``7/13/2018`` can only be
    month-first, so those are read without ceremony. ``7/3/2018`` is either the
    7th of March or the 3rd of July, and the two readings are four months
    apart. Guessing there would quietly shift account ages and posting dates by
    up to a year, so it is refused unless the caller states the order.
    """
    first, second, year = (int(match.group(index)) for index in (1, 2, 3))
    hour, minute, second_of = (int(match.group(i) or 0) for i in (4, 5, 6))

    if first > UNAMBIGUOUS_MONTH and second <= UNAMBIGUOUS_MONTH:
        day, month = first, second
    elif second > UNAMBIGUOUS_MONTH and first <= UNAMBIGUOUS_MONTH:
        month, day = first, second
    elif date_order is None:
        msg = (
            f"ambiguous date {match.group(0)!r}: it is either day/month or "
            "month/day, and the two readings differ by months. Pass "
            "date_order='dmy' or 'mdy' to state which this file uses."
        )
        raise AmbiguousDateError(msg)
    elif date_order == "dmy":
        day, month = first, second
    else:
        month, day = first, second
    return datetime(year, month, day, hour, minute, second_of)  # noqa: DTZ001 - caller attaches


def _parse_common_formats(raw: str, date_order: DateOrder | None = None) -> datetime:
    """Fall back to the timestamp layouts the archives actually ship."""
    slash = _SLASH_DATE.match(raw.strip())
    if slash:
        return _parse_slash_date(slash, date_order)
    for layout in _ISO_FORMATS:
        try:
            return datetime.strptime(raw, layout)  # noqa: DTZ007 - naiveté handled by caller
        except ValueError:
            continue
    msg = f"unrecognised timestamp format: {raw!r}"
    raise ValueError(msg)


_SCIENTIFIC_NOTATION = re.compile(r"^\d+(\.\d+)?[eE][+-]?\d+$")


def check_identifier(value: str, *, field: str) -> str:
    """Reject an identifier a spreadsheet has already destroyed.

    Raises:
        CorruptedIdentifierError: If the value is in scientific notation,
            which for an id means its digits are gone for good.
    """
    if _SCIENTIFIC_NOTATION.match(value.strip()):
        msg = (
            f"{field} {value!r} is in scientific notation: a spreadsheet has "
            "rounded this identifier and its digits cannot be recovered. It "
            "would join to nothing."
        )
        raise CorruptedIdentifierError(msg)
    return value


def _clean(value: object) -> object | None:
    """Turn empty CSV cells into ``None``, leave everything else alone."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _as_int(value: object) -> int | None:
    """Best-effort integer, ``None`` when absent or unparsable."""
    cleaned = _clean(value)
    if cleaned is None:
        return None
    try:
        return int(float(str(cleaned)))
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool | None:
    """Best-effort boolean over the spellings exports actually use."""
    cleaned = _clean(value)
    if cleaned is None:
        return None
    if isinstance(cleaned, bool):
        return cleaned
    text = str(cleaned).strip().casefold()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    return None


def _as_list(value: object, separator: str) -> tuple[str, ...]:
    """Read a list column: a JSON array, a bracketed list, a delimited string.

    The bracketed form matters: the Twitter Information Operations Archive
    writes hashtags and urls as ``[tag1, tag2]``, which is not valid JSON
    because the items are unquoted. Falling back to a split rather than
    returning nothing is the difference between reading those columns and
    silently dropping every hashtag in the archive.
    """
    cleaned = _clean(value)
    if cleaned is None:
        return ()
    if isinstance(cleaned, (list, tuple)):
        return tuple(str(item) for item in cleaned)
    text = str(cleaned)
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            inner = text.strip()[1:-1]
            return tuple(part.strip().strip("'\"") for part in inner.split(",") if part.strip())
        if isinstance(decoded, list):
            return tuple(str(item) for item in decoded)
        return ()
    return tuple(part.strip() for part in text.split(separator) if part.strip())


def _merge_extra(mapped: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    """Collect the columns the schema has no field for.

    A file written by :func:`write_corpus` already carries an ``extra`` object;
    it is merged rather than nested, so a corpus can be cached and re-read any
    number of times without growing an onion of ``extra.extra.extra``.
    """
    incoming = mapped.get("extra")
    extra: dict[str, Any] = dict(incoming) if isinstance(incoming, Mapping) else {}
    extra.update(
        {key: value for key, value in mapped.items() if key not in known and key != "extra"}
    )
    return extra


class NativeAdapter:
    """Reads SynthWatch's own schema from CSV, JSON and JSONL.

    Args:
        platform: Platform stamped on records that do not carry their own
            ``platform`` column.
        assume_timezone: Timezone for naive timestamps. ``None`` refuses them.
        date_order: How to read slash dates such as ``7/3/2018``. ``None``
            refuses the ambiguous ones.
        post_aliases: Extra source-column to schema-field mappings for posts,
            merged over :data:`DEFAULT_POST_ALIASES`.
        account_aliases: The same, for accounts.
        list_separator: Delimiter for list columns that are not JSON arrays.
        strict: Raise on the first bad row instead of counting it. Useful for
            a small curated file, wrong for a large archive.
    """

    name = "native"
    platform = Platform.GENERIC

    def __init__(
        self,
        *,
        platform: Platform = Platform.GENERIC,
        assume_timezone: tzinfo | None = None,
        date_order: DateOrder | None = None,
        post_aliases: Mapping[str, str] | None = None,
        account_aliases: Mapping[str, str] | None = None,
        list_separator: str = "|",
        strict: bool = False,
        post_row_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.platform = platform
        self.assume_timezone = assume_timezone
        self.date_order = date_order
        self.post_aliases = {**DEFAULT_POST_ALIASES, **(post_aliases or {})}
        self.account_aliases = {**DEFAULT_ACCOUNT_ALIASES, **(account_aliases or {})}
        self.list_separator = list_separator
        self.strict = strict
        self.post_row_transform = post_row_transform
        """Hook for platform adapters: rewrite a raw row before it is mapped.

        Aliases rename columns; this derives them. A source that encodes the
        conversational role as ``is_retweet`` plus ``in_reply_to_tweetid``
        needs a rule, not a rename, and that rule belongs to the adapter that
        knows the platform rather than to the schema.
        """

    # -- entry points ----------------------------------------------------

    def sniff(self, path: Path) -> bool:
        """Whether the suffix is one this adapter reads."""
        return path.suffix.casefold() in (JSON_SUFFIXES | JSONL_SUFFIXES | TABLE_SUFFIXES)

    def load(self, path: Path) -> LoadResult:
        """Read ``path`` into a corpus.

        A ``.json`` file may hold the full ``{accounts, posts, labels}``
        object or a bare array of posts; ``.jsonl`` and ``.csv`` are read as
        posts. Load a separate account table with :meth:`load_tables`.
        """
        suffix = path.suffix.casefold()
        if suffix in JSON_SUFFIXES:
            return self._load_json(path)
        if suffix in JSONL_SUFFIXES:
            return self._build(path.name, self._read_jsonl(path), (), ())
        if suffix in TABLE_SUFFIXES:
            return self._build(path.name, self._read_table(path), (), ())
        msg = f"unsupported file type {path.suffix!r}; expected .json, .jsonl, .csv or .tsv"
        raise ValueError(msg)

    def load_tables(
        self,
        posts: Path,
        accounts: Path | None = None,
        labels: Path | None = None,
    ) -> LoadResult:
        """Read posts, accounts and labels from separate files."""
        source = posts.name if accounts is None else f"{posts.name}+{accounts.name}"
        return self.from_records(
            source,
            self._read_any(posts),
            self._read_any(accounts) if accounts else (),
            self._read_any(labels) if labels else (),
        )

    def read_records(self, path: Path) -> Sequence[Mapping[str, Any]]:
        """Read a file into raw rows, without mapping them onto the schema.

        The seam a platform adapter builds on: read once, rearrange, then hand
        the rows back through :meth:`from_records`.
        """
        return self._read_any(path)

    def from_records(
        self,
        source: str,
        post_rows: Iterable[Mapping[str, Any]],
        account_rows: Iterable[Mapping[str, Any]] = (),
        label_rows: Iterable[Mapping[str, Any]] = (),
    ) -> LoadResult:
        """Map already-read rows onto the schema, counting every failure.

        Platform adapters use this rather than reimplementing the mapping, so
        that the drop-rate accounting is the same everywhere.
        """
        return self._build(source, post_rows, account_rows, label_rows)

    # -- readers ---------------------------------------------------------

    def _read_any(self, path: Path) -> Sequence[Mapping[str, Any]]:
        """Read a file of records, whatever of the three formats it is in."""
        suffix = path.suffix.casefold()
        if suffix in JSON_SUFFIXES:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
            msg = f"{path.name}: expected a JSON array of records"
            raise ValueError(msg)
        if suffix in JSONL_SUFFIXES:
            return self._read_jsonl(path)
        return self._read_table(path)

    def _load_json(self, path: Path) -> LoadResult:
        """Read a ``.json`` file that may be a corpus object or a post array."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return self._build(path.name, payload, (), ())
        if not isinstance(payload, dict):
            msg = f"{path.name}: expected an object or an array at the top level"
            raise ValueError(msg)
        return self._build(
            path.name,
            payload.get("posts", []),
            payload.get("accounts", []),
            payload.get("labels", []),
        )

    def _read_jsonl(self, path: Path) -> list[Mapping[str, Any]]:
        """Read one JSON object per line, ignoring blank lines."""
        records: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    if self.strict:
                        msg = f"{path.name}:{number}: {error}"
                        raise ValueError(msg) from error
                    records.append({"__parse_error__": str(error)})
        return records

    def _read_table(self, path: Path) -> list[Mapping[str, Any]]:
        """Read a CSV or TSV table, keeping every cell as text."""
        delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter, restkey=OVERFLOW_KEY))

    # -- mapping ---------------------------------------------------------

    def _rename(self, row: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
        """Apply the alias table, without letting an alias shadow a real column.

        A file carrying both ``id`` and ``post_id`` keeps ``post_id``: the
        canonical name always wins, whatever order the columns appear in.
        """
        canonical: dict[str, Any] = {}
        aliased: dict[str, Any] = {}
        for key, value in row.items():
            name = key.strip()
            target = aliases.get(name.casefold())
            if target is None:
                canonical[name] = value
            else:
                aliased.setdefault(target, value)
        for target, value in aliased.items():
            if _clean(canonical.get(target)) is None:
                canonical[target] = value
        return canonical

    def _post_from_row(self, row: Mapping[str, Any]) -> Post:
        """Build a :class:`Post`, routing unknown columns into ``extra``."""
        if self.post_row_transform is not None:
            row = self.post_row_transform(row)
        mapped = self._rename(row, self.post_aliases)
        known = set(Post.model_fields) - {"extra"}
        payload: dict[str, Any] = {
            key: _clean(value) for key, value in mapped.items() if key in known
        }
        extra = _merge_extra(mapped, known)

        payload["created_at"] = parse_timestamp(
            mapped.get("created_at", ""),
            assume_timezone=self.assume_timezone,
            date_order=self.date_order,
        )
        if mapped.get("collected_at"):
            payload["collected_at"] = parse_timestamp(
                mapped["collected_at"],
                assume_timezone=self.assume_timezone,
                date_order=self.date_order,
            )
        payload["platform"] = self._platform_of(mapped)
        for name in ("post_id", "account_id"):
            if payload.get(name) is not None:
                check_identifier(str(payload[name]), field=name)
        payload["text"] = str(_clean(mapped.get("text")) or "")
        for name in _POST_INT_FIELDS:
            if name in payload:
                payload[name] = _as_int(payload[name])
        payload["media_count"] = payload.get("media_count") or 0
        for name in _POST_LIST_FIELDS:
            payload[name] = _as_list(mapped.get(name), self.list_separator)
        if extra:
            payload["extra"] = extra
        return Post.model_validate(payload)

    def _account_from_row(self, row: Mapping[str, Any]) -> Account:
        """Build an :class:`Account`, routing unknown columns into ``extra``."""
        mapped = self._rename(row, self.account_aliases)
        known = set(Account.model_fields) - {"extra"}
        payload: dict[str, Any] = {
            key: _clean(value) for key, value in mapped.items() if key in known
        }
        extra = _merge_extra(mapped, known)

        for name in ("created_at", "collected_at"):
            if mapped.get(name):
                payload[name] = parse_timestamp(
                    mapped[name],
                    assume_timezone=self.assume_timezone,
                    date_order=self.date_order,
                )
            else:
                payload.pop(name, None)
        payload["platform"] = self._platform_of(mapped)
        if payload.get("account_id") is not None:
            check_identifier(str(payload["account_id"]), field="account_id")
        for name in _ACCOUNT_INT_FIELDS:
            if name in payload:
                payload[name] = _as_int(payload[name])
        for name in _ACCOUNT_BOOL_FIELDS:
            if name in payload:
                payload[name] = _as_bool(payload[name])
        if extra:
            payload["extra"] = extra
        return Account.model_validate(payload)

    def _label_from_row(self, row: Mapping[str, Any]) -> LabelRecord:
        """Build a :class:`LabelRecord`; provenance columns are required."""
        payload = {key: _clean(value) for key, value in row.items()}
        payload["platform"] = self._platform_of(row)
        if payload.get("labelled_at"):
            payload["labelled_at"] = parse_timestamp(
                payload["labelled_at"],
                assume_timezone=self.assume_timezone,
                date_order=self.date_order,
            )
        return LabelRecord.model_validate(payload)

    def _platform_of(self, row: Mapping[str, Any]) -> Platform:
        """Per-row platform when the file declares one, else the adapter default."""
        declared = _clean(row.get("platform"))
        if declared is None:
            return self.platform
        try:
            return Platform(str(declared).strip().casefold())
        except ValueError:
            return self.platform

    # -- assembly --------------------------------------------------------

    def _build(
        self,
        source: str,
        post_rows: Iterable[Mapping[str, Any]],
        account_rows: Iterable[Mapping[str, Any]],
        label_rows: Iterable[Mapping[str, Any]],
    ) -> LoadResult:
        """Map every row, counting failures instead of losing them."""
        skipped: Counter[str] = Counter()
        warnings: list[str] = []

        posts = list(self._map_rows(post_rows, self._post_from_row, skipped, "post"))
        accounts = list(self._map_rows(account_rows, self._account_from_row, skipped, "account"))
        labels = list(self._map_rows(label_rows, self._label_from_row, skipped, "label"))

        corpus = Corpus(
            accounts=tuple(accounts),
            posts=tuple(posts),
            labels=tuple(labels),
            source=source,
        )
        if corpus.orphan_post_ids:
            warnings.append(
                f"{len(corpus.orphan_post_ids)} post(s) reference accounts absent from "
                "this load; they stay in the corpus and are reported by orphan_post_ids"
            )
        total = len(posts) + skipped.total()
        if total and skipped.total() / total > HIGH_DROP_RATE:
            warnings.append(
                f"{skipped.total() / total:.0%} of rows were skipped; the drop rate is "
                "itself a finding and belongs in the report"
            )
        return LoadResult(
            corpus=corpus,
            report=IngestReport(
                source=source,
                n_accounts=len(accounts),
                n_posts=len(posts),
                n_skipped=skipped.total(),
                skip_reasons=dict(skipped),
                warnings=tuple(warnings),
            ),
        )

    def _map_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        build: Callable[[Mapping[str, Any]], _Record],
        skipped: Counter[str],
        kind: str,
    ) -> Iterator[_Record]:
        """Apply ``build`` to each row, recording why any of them failed."""
        for row in rows:
            if "__parse_error__" in row:
                skipped[f"{kind}:malformed_json"] += 1
                continue
            if OVERFLOW_KEY in row:
                skipped[f"{kind}:ragged_row"] += 1
                continue
            try:
                yield build(row)
            except (ValueError, TypeError, KeyError) as error:
                if self.strict:
                    raise
                skipped[f"{kind}:{_skip_reason(error)}"] += 1


_SKIP_REASONS: Final[Mapping[type[Exception], str]] = {
    NaiveTimestampError: "naive_timestamp",
    AmbiguousDateError: "ambiguous_date",
    CorruptedIdentifierError: "corrupted_identifier",
}
"""Named reasons for the failures a report should be able to act on."""


def _skip_reason(error: Exception) -> str:
    """Compress a failure into a countable reason string."""
    for kind, reason in _SKIP_REASONS.items():
        if isinstance(error, kind):
            return reason
    if isinstance(error, ValidationError):
        return _validation_reason(error)
    return type(error).__name__


def _validation_reason(error: ValidationError) -> str:
    """Compress a pydantic error into a countable reason string."""
    first = error.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "record"
    return f"{location}:{first['type']}"


def write_corpus(corpus: Corpus, path: Path) -> Path:
    """Write ``corpus`` as a native JSON file that :meth:`load` reads back.

    Caching a normalised corpus is what makes an analysis re-runnable without
    the original export, which may be large, rate-limited or gone.
    """
    payload = {
        "source": corpus.source,
        "collected_at": corpus.collected_at.isoformat() if corpus.collected_at else None,
        "accounts": [account.model_dump(mode="json") for account in corpus.accounts],
        "posts": [post.model_dump(mode="json") for post in corpus.posts],
        "labels": [label.model_dump(mode="json") for label in corpus.labels],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def timezone_of(offset_hours: float) -> tzinfo:
    """Build a fixed-offset timezone, for ``assume_timezone`` at a call site."""
    return timezone(timedelta(hours=offset_hours))


REGISTRY.register(NativeAdapter())
