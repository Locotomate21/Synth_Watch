"""Shared enumerations and type aliases for the internal schema.

Identifiers are plain strings rather than ``NewType``s so that adapters and
tests can build objects from raw payloads without ceremony. They are always
*platform-scoped*: two accounts from different platforms may legitimately
share the same ``account_id``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

AccountId: TypeAlias = str
PostId: TypeAlias = str
ClusterId: TypeAlias = str


class Platform(StrEnum):
    """Source platform a record was collected from."""

    REDDIT = "reddit"
    BLUESKY = "bluesky"
    MASTODON = "mastodon"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    GENERIC = "generic"
    """Data loaded from the native CSV/JSON schema without a known platform."""


class PostKind(StrEnum):
    """Conversational role of a post.

    Kept small on purpose: every platform can map onto these four without
    inventing a fifth. Platform-specific nuance goes into ``Post.extra``.
    """

    ORIGINAL = "original"
    REPLY = "reply"
    REPOST = "repost"
    QUOTE = "quote"


class Label(StrEnum):
    """Ground-truth class used for supervised training and evaluation.

    These describe *account behaviour as recorded by a labelled dataset*, not a
    verdict produced by this library. ``synthwatch.detect`` emits calibrated
    probabilities; it never assigns a :class:`Label`.
    """

    AUTOMATED = "automated"
    """Scripted or scheduled posting (Bot Repository style labels)."""

    INFO_OPERATION = "info_operation"
    """Attributed to a state-linked influence operation (e.g. the Twitter
    Information Operations Archive). Not a synonym of ``AUTOMATED``: many such
    accounts were operated by humans."""

    ORGANIC = "organic"
    """Labelled as a regular human account by the source dataset."""

    UNKNOWN = "unknown"


class LabelMethod(StrEnum):
    """How a label was produced. Recorded so that label noise stays auditable."""

    HUMAN_ANNOTATION = "human_annotation"
    PLATFORM_ENFORCEMENT = "platform_enforcement"
    """Derived from a suspension/takedown. Reflects a platform's decision, with
    its own error rate and policy bias."""

    HEURISTIC = "heuristic"
    SELF_DECLARED = "self_declared"
    """Account self-identifies as a bot (bio, ``bot`` flag, app attribution)."""

    UNKNOWN = "unknown"


class FeatureLevel(StrEnum):
    """Unit of analysis a feature is computed over."""

    POST = "post"
    ACCOUNT = "account"
    CLUSTER = "cluster"
