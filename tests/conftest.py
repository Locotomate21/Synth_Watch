"""Synthetic fixtures shared across the test suite.

Everything here is generated, never collected: no real account, handle or post
appears in this repository. The factories are deterministic given a seed so
that a failing coordination test can be reproduced exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from synthwatch.models import Account, Corpus, Post, build_corpus
from synthwatch.types import Platform, PostKind

EPOCH = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
"""Fixed reference instant for every synthetic fixture."""


def make_account(
    account_id: str,
    *,
    platform: Platform = Platform.GENERIC,
    created_days_ago: float = 900.0,
    followers: int | None = 340,
    following: int | None = 280,
    **overrides: object,
) -> Account:
    """Build a synthetic account with plausible, unremarkable defaults.

    Any field can be overridden, including back to ``None``: the defaults are a
    dict the caller updates rather than fixed keyword arguments, so
    ``make_account("a1", handle=None)`` builds an account with no handle instead
    of raising a duplicate-argument error.
    """
    defaults: dict[str, object] = {
        "account_id": account_id,
        "platform": platform,
        "handle": f"user_{account_id}",
        "display_name": f"Test Account {account_id}",
        "created_at": EPOCH - timedelta(days=created_days_ago),
        "followers_count": followers,
        "following_count": following,
        "collected_at": EPOCH,
    }
    return Account(**{**defaults, **overrides})


def make_post(
    post_id: str,
    account_id: str,
    *,
    offset_minutes: float = 0.0,
    text: str = "",
    platform: Platform = Platform.GENERIC,
    kind: PostKind = PostKind.ORIGINAL,
    **overrides: object,
) -> Post:
    """Build a synthetic post ``offset_minutes`` after :data:`EPOCH`."""
    defaults: dict[str, object] = {
        "post_id": post_id,
        "account_id": account_id,
        "platform": platform,
        "created_at": EPOCH + timedelta(minutes=offset_minutes),
        "text": text,
        "kind": kind,
        "collected_at": EPOCH,
    }
    return Post(**{**defaults, **overrides})


def make_corpus(posts: Sequence[Post], *, source: str = "synthetic") -> Corpus:
    """Wrap ``posts`` in a corpus, inventing an account for every author."""
    accounts = [make_account(account_id) for account_id in sorted({p.account_id for p in posts})]
    return build_corpus(accounts, posts, source=source, collected_at=EPOCH)


@pytest.fixture
def epoch() -> datetime:
    """The fixed reference instant used by the factories."""
    return EPOCH


@pytest.fixture
def quiet_corpus() -> Corpus:
    """Three unrelated accounts posting different things at different times."""
    posts = [
        make_post("p1", "a1", offset_minutes=0, text="El debate de anoche fue largo."),
        make_post("p2", "a2", offset_minutes=137, text="Mi gato rompio otra taza hoy."),
        make_post("p3", "a3", offset_minutes=901, text="Alguien sabe si abre la biblioteca?"),
        make_post("p4", "a1", offset_minutes=1500, text="Se me hizo tarde otra vez."),
    ]
    return make_corpus(posts, source="synthetic:quiet")
