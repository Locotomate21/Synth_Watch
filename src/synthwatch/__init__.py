"""SynthWatch: measuring automation and synthetic text in political conversation.

The library is organised in three layers:

``synthwatch.ingest``
    Pluggable adapters that normalise CSV/JSON exports, Reddit, Bluesky and
    Mastodon data into one internal schema.
``synthwatch.detect``
    Feature extractors (text, account, temporal, coordination) and a calibrated
    ensemble over them.
``synthwatch.report``
    Aggregate metrics, cluster cards and JSON/HTML export.

What this library does not do: label an individual account as a bot. Outputs
are calibrated probabilities with stated uncertainty, aggregated to the level
of a cluster or a corpus. See ``ETHICS.md``.
"""

from __future__ import annotations

from synthwatch.models import Account, Corpus, LabelRecord, Post, build_corpus
from synthwatch.types import (
    AccountId,
    ClusterId,
    FeatureLevel,
    Label,
    LabelMethod,
    Platform,
    PostId,
    PostKind,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Account",
    "AccountId",
    "ClusterId",
    "Corpus",
    "FeatureLevel",
    "Label",
    "LabelMethod",
    "LabelRecord",
    "Platform",
    "Post",
    "PostId",
    "PostKind",
    "__version__",
    "build_corpus",
]
