"""Synthetic posting rhythms for the temporal module.

Four archetypes, each planted so a test can assert what the features must say
about it: someone with a sleep cycle, a metronome, a queue that discharges in
clumps, and a source that only records dates.

Timestamps are anchored to midnight UTC and built from a seeded generator, so
a failing assertion is reproducible exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from random import Random

from synthwatch.models import Corpus, Post
from synthwatch.types import Platform, PostKind
from tests.conftest import make_corpus

ANCHOR = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
"""Midnight UTC, so hour-of-day assertions read directly off the offsets."""

AWAKE_START = 8.0
AWAKE_END = 22.5
"""The waking window the synthetic human posts inside, in UTC hours."""


def post_at(post_id: str, account_id: str, when: datetime, **overrides: object) -> Post:
    """A post at an exact instant, for rhythms that care about the clock."""
    return Post(
        post_id=post_id,
        account_id=account_id,
        platform=Platform.GENERIC,
        created_at=when,
        text="Un mensaje cualquiera con longitud suficiente para el analisis.",
        collected_at=ANCHOR + timedelta(days=90),
        **overrides,
    )


def human_posts(
    account_id: str = "human",
    *,
    days: int = 40,
    day_off_probability: float = 0.25,
    seed: int = 7,
) -> list[Post]:
    """Someone who sleeps, takes days off, and posts in irregular clumps."""
    rng = Random(seed)
    posts: list[Post] = []
    for day in range(days):
        if rng.random() < day_off_probability:
            continue
        for index in range(rng.randint(1, 5)):
            hour = rng.uniform(AWAKE_START, AWAKE_END)
            when = ANCHOR + timedelta(days=day, hours=hour, seconds=rng.uniform(0, 59))
            posts.append(post_at(f"{account_id}_d{day}_{index}", account_id, when))
    return posts


def metronome_posts(
    account_id: str = "metro",
    *,
    days: int = 10,
    period_minutes: float = 45.0,
) -> list[Post]:
    """A fixed cadence, round the clock, never missing a day."""
    total = int(days * 24 * 60 / period_minutes)
    return [
        post_at(
            f"{account_id}_{index}",
            account_id,
            ANCHOR + timedelta(minutes=index * period_minutes),
        )
        for index in range(total)
    ]


def bursty_posts(
    account_id: str = "bursty",
    *,
    sessions: int = 10,
    per_session: int = 6,
    session_gap_days: float = 2.0,
    intra_seconds: float = 15.0,
    drift_hours: float = 3.7,
) -> list[Post]:
    """A queue that discharges in clumps, then goes silent for days.

    Sessions drift around the clock rather than firing at the same hour, so the
    fixture exercises burst detection without accidentally being a degenerate
    circadian histogram as well.
    """
    return [
        post_at(
            f"{account_id}_s{session}_{index}",
            account_id,
            ANCHOR
            + timedelta(
                days=session * session_gap_days,
                hours=session * drift_hours,
                seconds=index * intra_seconds,
            ),
        )
        for session in range(sessions)
        for index in range(per_session)
    ]


def coarse_posts(account_id: str = "coarse", *, days: int = 30) -> list[Post]:
    """A source that records dates only: every timestamp lands on midnight."""
    return [
        post_at(f"{account_id}_{day}", account_id, ANCHOR + timedelta(days=day))
        for day in range(days)
    ]


def sparse_posts(account_id: str = "sparse", *, count: int = 5) -> list[Post]:
    """Too few posts to describe a rhythm at all."""
    return [
        post_at(f"{account_id}_{index}", account_id, ANCHOR + timedelta(days=index * 3))
        for index in range(count)
    ]


NIGHT_HOURS = (1.0, 3.0, 5.0)
ORIGINAL_HOURS = (9.0, 13.0, 17.0, 21.0)


def repost_heavy_posts(account_id: str = "amplifier", *, days: int = 30) -> list[Post]:
    """Someone who writes by day while a tool reshares through their night.

    With reposts counted the account has no quiet window; with reposts excluded
    it sleeps like anyone else. The two readings are both correct, which is why
    ``include_reposts`` is a documented choice rather than a default nobody
    sees.
    """
    posts: list[Post] = []
    for day in range(days):
        for hour in ORIGINAL_HOURS:
            posts.append(
                post_at(
                    f"{account_id}_d{day}_o{hour:.0f}",
                    account_id,
                    ANCHOR + timedelta(days=day, hours=hour),
                )
            )
        for hour in NIGHT_HOURS:
            posts.append(
                post_at(
                    f"{account_id}_d{day}_r{hour:.0f}",
                    account_id,
                    ANCHOR + timedelta(days=day, hours=hour),
                    kind=PostKind.REPOST,
                )
            )
    return posts


def shifted(posts: Sequence[Post], hours: float) -> list[Post]:
    """The same timeline, rotated around the clock.

    Every feature in ``detect.temporal`` must be blind to this, since the
    corpus never says which timezone an account lives in.
    """
    return [
        post.model_copy(update={"created_at": post.created_at + timedelta(hours=hours)})
        for post in posts
    ]


def rhythm_corpus() -> Corpus:
    """All four archetypes in one corpus."""
    return make_corpus(
        [*human_posts(), *metronome_posts(), *bursty_posts(), *sparse_posts()],
        source="synthetic:rhythms",
    )
