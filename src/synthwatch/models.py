"""The internal schema shared by every SynthWatch module.

Everything the library reasons about is an :class:`Account`, a :class:`Post`,
or a :class:`Corpus` of both.

Design rules for this module:

* **Adapters normalise, models validate.** Platform quirks are resolved in
  ``synthwatch.ingest``; whatever survives here is comparable across platforms.
  Payload fields with no schema equivalent live in ``extra``, so nothing is
  silently discarded.
* **Models are dumb.** No feature is computed here. Any derived quantity that
  encodes a modelling choice belongs in ``synthwatch.detect``.
* **No verdict fields.** An ``Account`` has no ``is_bot`` attribute and never
  will. Ground truth is attached out of band through :class:`LabelRecord`,
  always with its provenance; predictions are returned by ``detect`` as
  calibrated probabilities, never written back onto the record.
* **Time is explicit.** Naive datetimes are rejected instead of assumed to be
  UTC: circadian and inter-arrival features are worthless if the offset was
  guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synthwatch.types import AccountId, Label, LabelMethod, Platform, PostId, PostKind


def require_utc(value: datetime) -> datetime:
    """Normalise an aware datetime to UTC, rejecting naive ones.

    Args:
        value: A timezone-aware datetime.

    Returns:
        The same instant expressed in UTC.

    Raises:
        ValueError: If ``value`` carries no usable UTC offset.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = (
            "naive datetime rejected: attach a timezone at ingest time. "
            "Silently assuming UTC corrupts circadian and burst features."
        )
        raise ValueError(msg)
    return value.astimezone(UTC)


def optional_utc(value: datetime | None) -> datetime | None:
    """Apply :func:`require_utc` to a value that may be ``None``."""
    return None if value is None else require_utc(value)


class _Record(BaseModel):
    """Base class for immutable schema records.

    Note:
        ``frozen=True`` makes instances read-only but *not* hashable, because
        ``extra`` is a ``dict``. Use ``.key`` as a set or dict key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    extra: dict[str, Any] = Field(default_factory=dict)
    """Source fields with no schema equivalent, kept verbatim for auditability."""


class Account(_Record):
    """An account as observed at collection time.

    Counts are snapshots, not time series: a corpus crawled months after the
    posts it contains carries follower numbers that never existed while those
    posts were written. ``collected_at`` is what makes that gap auditable
    instead of invisible.
    """

    account_id: AccountId
    platform: Platform
    handle: str | None = None
    display_name: str | None = None
    created_at: datetime | None = None
    description: str | None = None
    location: str | None = None
    url: str | None = None
    language: str | None = None

    followers_count: int | None = Field(default=None, ge=0)
    following_count: int | None = Field(default=None, ge=0)
    post_count: int | None = Field(default=None, ge=0)

    verified: bool | None = None
    protected: bool | None = None
    has_default_profile_image: bool | None = None

    declared_automated: bool | None = None
    """Platform-declared automation flag (Mastodon ``bot``, Reddit app posts,
    a self-identifying bio). This is a *label source*, never a model feature:
    training on it would only teach the ensemble to find the honest bots."""

    collected_at: datetime | None = None

    @field_validator("created_at", "collected_at")
    @classmethod
    def _normalise_timestamps(cls, value: datetime | None) -> datetime | None:
        return optional_utc(value)

    @field_validator("handle")
    @classmethod
    def _normalise_handle(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lstrip("@")
        return cleaned or None

    @property
    def key(self) -> tuple[Platform, AccountId]:
        """Hashable identity, unique across platforms."""
        return (self.platform, self.account_id)

    def age_days(self, reference: datetime) -> float | None:
        """Account age in days at ``reference``, or ``None`` if unknown.

        Args:
            reference: Aware datetime to measure the age against, normally the
                collection time rather than "now" -- a corpus re-analysed a
                year later must not silently age every account in it.
        """
        if self.created_at is None:
            return None
        return (require_utc(reference) - self.created_at).total_seconds() / 86_400.0


class Post(_Record):
    """A single message, comment, repost or quote.

    ``text`` is stored verbatim: no case folding, no whitespace collapsing, no
    Unicode normalisation. Typo rate, sentence-length uniformity and SimHash
    near-duplicate detection each depend on the original characters, so
    normalisation happens per feature and is documented there.
    """

    post_id: PostId
    account_id: AccountId
    platform: Platform
    created_at: datetime
    text: str = ""

    language: str | None = None
    kind: PostKind = PostKind.ORIGINAL

    parent_post_id: PostId | None = None
    root_post_id: PostId | None = None
    """Conversation root. Used to avoid reading a single busy thread as
    coordination between the people replying inside it."""

    urls: tuple[str, ...] = ()
    hashtags: tuple[str, ...] = ()
    mentions: tuple[AccountId, ...] = ()
    media_count: int = Field(default=0, ge=0)

    like_count: int | None = Field(default=None, ge=0)
    repost_count: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)

    client: str | None = None
    """Posting client or app, where the platform exposes it."""

    collected_at: datetime | None = None

    @field_validator("created_at")
    @classmethod
    def _normalise_created_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator("collected_at")
    @classmethod
    def _normalise_collected_at(cls, value: datetime | None) -> datetime | None:
        return optional_utc(value)

    @property
    def key(self) -> tuple[Platform, PostId]:
        """Hashable identity, unique across platforms."""
        return (self.platform, self.post_id)

    @property
    def is_reply(self) -> bool:
        """Whether this post replies to another post."""
        return self.kind is PostKind.REPLY

    @property
    def is_repost(self) -> bool:
        """Whether this post carries no original text of its own."""
        return self.kind is PostKind.REPOST


class LabelRecord(BaseModel):
    """Ground truth about an account, with the provenance that qualifies it.

    Provenance is mandatory. A suspension-derived label and a hand-annotated
    label disagree in systematic ways, and an evaluation that mixes them
    without saying so is not reproducible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_id: AccountId
    platform: Platform
    label: Label
    dataset: str
    """Dataset identifier, e.g. ``"indiana-bot-repository/cresci-2017"``."""

    method: LabelMethod = LabelMethod.UNKNOWN
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    labelled_at: datetime | None = None
    notes: str | None = None

    @field_validator("labelled_at")
    @classmethod
    def _normalise_timestamps(cls, value: datetime | None) -> datetime | None:
        return optional_utc(value)


@dataclass
class Corpus:
    """A bundle of accounts, posts and optional labels, immutable by convention.

    Posts are sorted by ``(created_at, post_id)`` at construction so every
    downstream module -- above all the sliding windows in
    ``detect.coordination`` -- sees one canonical ordering.

    Referential integrity is *reported*, not enforced: real collections contain
    posts whose author was suspended before the profile crawl reached them. Use
    :attr:`orphan_post_ids` to decide what to do about that, or
    :func:`build_corpus` with ``strict=True`` to refuse.
    """

    accounts: tuple[Account, ...] = ()
    posts: tuple[Post, ...] = ()
    labels: tuple[LabelRecord, ...] = ()
    source: str | None = None
    collected_at: datetime | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Freeze the sequences and impose the canonical post ordering."""
        self.accounts = tuple(self.accounts)
        self.posts = tuple(sorted(self.posts, key=lambda p: (p.created_at, p.post_id)))
        self.labels = tuple(self.labels)
        self.collected_at = optional_utc(self.collected_at)

    # -- indices ---------------------------------------------------------

    @cached_property
    def accounts_by_id(self) -> dict[AccountId, Account]:
        """Account lookup by id; on duplicates the last record wins."""
        return {account.account_id: account for account in self.accounts}

    @cached_property
    def posts_by_account(self) -> dict[AccountId, tuple[Post, ...]]:
        """Posts grouped by author, each group in chronological order."""
        grouped: dict[AccountId, list[Post]] = {}
        for post in self.posts:
            grouped.setdefault(post.account_id, []).append(post)
        return {account_id: tuple(items) for account_id, items in grouped.items()}

    @cached_property
    def labels_by_account(self) -> dict[AccountId, LabelRecord]:
        """Label lookup by account id; on duplicates the last record wins."""
        return {record.account_id: record for record in self.labels}

    # -- integrity -------------------------------------------------------

    @cached_property
    def orphan_post_ids(self) -> tuple[PostId, ...]:
        """Ids of posts whose author is missing from :attr:`accounts`."""
        known = self.accounts_by_id
        return tuple(p.post_id for p in self.posts if p.account_id not in known)

    @cached_property
    def silent_account_ids(self) -> tuple[AccountId, ...]:
        """Ids of accounts with no posts in this corpus."""
        authors = self.posts_by_account
        return tuple(a.account_id for a in self.accounts if a.account_id not in authors)

    @property
    def time_span(self) -> tuple[datetime, datetime] | None:
        """First and last post timestamp, or ``None`` if there are no posts."""
        if not self.posts:
            return None
        return (self.posts[0].created_at, self.posts[-1].created_at)

    # -- derivation ------------------------------------------------------

    def filter_posts(self, predicate: Callable[[Post], bool]) -> Self:
        """Return a corpus keeping only posts matching ``predicate``.

        Accounts and labels are carried over untouched, so an account that
        loses all of its posts becomes a silent account rather than vanishing.
        """
        return type(self)(
            accounts=self.accounts,
            posts=tuple(p for p in self.posts if predicate(p)),
            labels=self.labels,
            source=self.source,
            collected_at=self.collected_at,
            notes=self.notes,
        )

    def subset(self, account_ids: Iterable[AccountId]) -> Self:
        """Return a corpus restricted to ``account_ids``."""
        wanted = set(account_ids)
        return type(self)(
            accounts=tuple(a for a in self.accounts if a.account_id in wanted),
            posts=tuple(p for p in self.posts if p.account_id in wanted),
            labels=tuple(r for r in self.labels if r.account_id in wanted),
            source=self.source,
            collected_at=self.collected_at,
            notes=self.notes,
        )

    def merge(self, other: Corpus, *, source: str | None = None) -> Self:
        """Concatenate two corpora, de-duplicating records by platform and id."""
        accounts = {a.key: a for a in (*self.accounts, *other.accounts)}
        posts = {p.key: p for p in (*self.posts, *other.posts)}
        labels = {(r.platform, r.account_id, r.dataset): r for r in (*self.labels, *other.labels)}
        return type(self)(
            accounts=tuple(accounts.values()),
            posts=tuple(posts.values()),
            labels=tuple(labels.values()),
            source=source or self.source,
            collected_at=self.collected_at,
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate description of the corpus, for report headers and logs."""
        span = self.time_span
        return {
            "source": self.source,
            "n_accounts": len(self.accounts),
            "n_posts": len(self.posts),
            "n_labels": len(self.labels),
            "platforms": sorted({p.platform.value for p in self.posts}),
            "first_post": span[0].isoformat() if span else None,
            "last_post": span[1].isoformat() if span else None,
            "orphan_posts": len(self.orphan_post_ids),
            "silent_accounts": len(self.silent_account_ids),
        }

    def __len__(self) -> int:
        """Number of posts in the corpus."""
        return len(self.posts)


def build_corpus(
    accounts: Sequence[Account],
    posts: Sequence[Post],
    *,
    labels: Sequence[LabelRecord] = (),
    source: str | None = None,
    collected_at: datetime | None = None,
    strict: bool = False,
) -> Corpus:
    """Build a :class:`Corpus`, optionally refusing orphan posts.

    Args:
        accounts: Account records.
        posts: Post records.
        labels: Optional ground truth.
        source: Human-readable provenance string, e.g. a file name or query.
        collected_at: When the data was collected.
        strict: If ``True``, raise when a post references an unknown account.

    Returns:
        The assembled corpus.

    Raises:
        ValueError: If ``strict`` is set and orphan posts are present.
    """
    corpus = Corpus(
        accounts=tuple(accounts),
        posts=tuple(posts),
        labels=tuple(labels),
        source=source,
        collected_at=collected_at,
    )
    if strict and corpus.orphan_post_ids:
        preview = ", ".join(corpus.orphan_post_ids[:5])
        msg = f"{len(corpus.orphan_post_ids)} post(s) reference unknown accounts: {preview}"
        raise ValueError(msg)
    return corpus
