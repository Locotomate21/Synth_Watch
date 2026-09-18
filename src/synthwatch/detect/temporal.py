"""When an account posts: circadian shape, inter-arrival structure, bursts.

Every feature in this module is **invariant to a rotation of the clock**. That
is a deliberate constraint, not a coincidence: the schema records UTC, and a
corpus almost never says which timezone each account lives in. A feature that
needed local time would either require a guess -- which is wrong for a
predictable fraction of any real corpus -- or quietly restrict the analysis to
one country.

So instead of asking "does this account post at 3am local time?", the module
asks questions a rotation cannot change:

* How concentrated is the posting across the 24 hours of the day, whatever
  those hours are called locally? (:func:`circadian_entropy`)
* How much does the account post during *its own* quietest stretch? A person
  sleeps somewhere; a scheduler does not. (:func:`quiet_hours_share`)
* How regular are the gaps between posts? (:func:`interarrival_entropy`)
* Does activity arrive in bursts, relative to the account's own baseline?
  (:func:`find_bursts`)

Reposts count here, unlike in ``detect.coordination``. The timing of
amplification is behaviour: a queue that reshares on a fixed cadence is exactly
what this module should see.

Known limitations
-----------------

* **Volume drives everything.** An account with eight posts has no measurable
  circadian shape. Below ``min_posts`` the features are ``NaN`` rather than
  zero, and that threshold is a judgement call: at 20 posts the estimates are
  still noisy, they are simply no longer meaningless.
* **Scheduling is not automation.** Newsrooms, campaign accounts and anyone
  using a social media manager post round the clock on a regular cadence, and
  score exactly like a bot on every feature here.
* **Timestamp resolution varies by source.** Some archives truncate to the
  minute, some to the day. Day-resolution data makes inter-arrival features
  meaningless while still producing a number, so ``TemporalProfile`` reports
  the observed resolution and the extractor refuses the sub-daily features
  when timestamps are too coarse.
* **A shared quiet window is not a shared operator.** Two accounts in the same
  country will agree on their quiet hours for entirely ordinary reasons.
* **Rotation invariance is exact only for whole-hour offsets.** India (+5:30)
  and Nepal (+5:45) move posts across hour boundaries rather than rotating the
  histogram, so the circadian figure drifts slightly for accounts there. The
  drift is small -- a few percent of the normalised entropy -- but it is not
  zero, and a study centred on those countries should say so.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from statistics import median

import pandas as pd

from synthwatch.detect.base import FeatureSpec, empty_frame
from synthwatch.detect.stats import shannon_entropy
from synthwatch.models import Corpus, Post
from synthwatch.types import AccountId, FeatureLevel

__all__ = [
    "TemporalConfig",
    "TemporalExtractor",
    "TemporalProfile",
    "circadian_entropy",
    "find_bursts",
    "interarrival_entropy",
    "profile_accounts",
    "quiet_hours_share",
    "shannon_entropy",
]

HOURS_PER_DAY = 24
SECONDS_PER_DAY = 86_400.0

MIN_POSTS_FOR_GAPS = 2
"""One post has no gap to measure, so nothing below this describes a rhythm."""

GAP_BIN_COUNT = 21
"""Log2 bins from one second to about twelve days, which spans every realistic
inter-post gap without letting one enormous gap dominate the distribution."""


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def hour_histogram(times: Sequence[datetime]) -> list[int]:
    """Posts per hour of the day, in the 24 bins of the recorded timezone."""
    counts = [0] * HOURS_PER_DAY
    for moment in times:
        counts[moment.hour] += 1
    return counts


def circadian_entropy(times: Sequence[datetime]) -> float:
    """How evenly posting is spread across the 24 hours of the day.

    Rotating every timestamp by a fixed offset permutes the histogram
    cyclically and leaves this value unchanged, which is what makes it usable
    without knowing anyone's timezone.
    """
    return shannon_entropy(hour_histogram(times), bins=HOURS_PER_DAY)


def quiet_hours_share(times: Sequence[datetime], *, window_hours: int = 6) -> tuple[float, int]:
    """Share of posts falling in the account's own quietest contiguous window.

    Args:
        times: Post timestamps.
        window_hours: Width of the window, defaulting to six hours: shorter
            than a night's sleep, so an irregular sleeper still registers.

    Returns:
        The share of posts inside the quietest window, and the hour that
        window starts at. A person who sleeps scores near zero whatever their
        timezone; a round-the-clock scheduler approaches
        ``window_hours / 24``.
    """
    counts = hour_histogram(times)
    total = sum(counts)
    if total == 0:
        return 0.0, 0
    window = min(window_hours, HOURS_PER_DAY)
    sums = [
        (sum(counts[(start + offset) % HOURS_PER_DAY] for offset in range(window)), start)
        for start in range(HOURS_PER_DAY)
    ]
    quietest, start_hour = min(sums)
    return quietest / total, start_hour


def gaps_seconds(times: Sequence[datetime]) -> list[float]:
    """Gaps between consecutive posts, in seconds, assuming sorted input."""
    return [(later - earlier).total_seconds() for earlier, later in pairwise(times)]


def interarrival_entropy(gaps: Sequence[float]) -> float:
    """Normalised entropy of log-binned inter-post gaps.

    Gaps are bucketed by powers of two rather than linearly, because the
    interesting contrast is between orders of magnitude -- seconds against
    hours -- not between 61 and 62 minutes. A metronomic account puts every
    gap in one bucket and scores near zero.
    """
    if not gaps:
        return 0.0
    buckets: Counter[int] = Counter()
    for gap in gaps:
        index = int(math.log2(max(gap, 1.0)))
        buckets[min(index, GAP_BIN_COUNT - 1)] += 1
    return shannon_entropy(list(buckets.values()), bins=GAP_BIN_COUNT)


def find_bursts(
    times: Sequence[datetime],
    *,
    factor: float = 0.25,
    ceiling: timedelta = timedelta(hours=1),
    min_size: int = 3,
) -> list[tuple[int, int]]:
    """Find runs of posts arriving far faster than the account's own baseline.

    A burst is a maximal run of at least ``min_size`` posts whose consecutive
    gaps all fall below ``factor`` times the account's *mean* gap, capped at
    ``ceiling``. The threshold is self-relative so that a prolific account and
    a quiet one are each measured against their own rhythm; the ceiling stops
    a very slow account from having three posts in an afternoon called a
    burst.

    The baseline is the mean rather than the median deliberately. When most of
    an account's posts arrive inside bursts -- precisely the case this function
    exists to detect -- the median gap *is* the intra-burst gap, and a
    median-based threshold would find nothing. The mean keeps measuring the
    overall rate, which is what a burst is fast relative to.

    Returns:
        Half-open ``(start, end)`` index ranges into ``times``.
    """
    gaps = gaps_seconds(times)
    if len(gaps) < min_size - 1 or min_size < MIN_POSTS_FOR_GAPS:
        return []
    baseline = sum(gaps) / len(gaps)
    threshold = min(factor * baseline, ceiling.total_seconds())
    if threshold <= 0:
        return []

    runs: list[tuple[int, int]] = []
    start = 0
    for index, gap in enumerate(gaps):
        if gap > threshold:
            if index + 1 - start >= min_size:
                runs.append((start, index + 1))
            start = index + 1
    if len(times) - start >= min_size:
        runs.append((start, len(times)))
    return runs


# --------------------------------------------------------------------------
# Configuration and profiles
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Parameters of the temporal analysis.

    Attributes:
        min_posts: Below this many posts an account gets ``NaN`` rather than a
            number. Circadian shape estimated from a handful of posts is
            noise wearing a decimal point.
        quiet_window_hours: Width of the quiet window in
            :func:`quiet_hours_share`.
        burst_factor: Fraction of the account's median gap under which
            consecutive posts count as bursting.
        burst_ceiling: Absolute cap on the burst threshold, so a slow account
            cannot burst over the course of an afternoon.
        min_burst_size: Posts required to call a run a burst.
        include_reposts: Whether reshares count. They do by default: the
            timing of amplification is behaviour too.
        min_resolution_seconds: Coarsest timestamp resolution at which the
            inter-arrival and burst features are still computed. Day-resolution
            archives produce confident nonsense otherwise.
    """

    min_posts: int = 20
    quiet_window_hours: int = 6
    burst_factor: float = 0.25
    burst_ceiling: timedelta = timedelta(hours=1)
    min_burst_size: int = 3
    include_reposts: bool = True
    min_resolution_seconds: float = 3600.0

    def __post_init__(self) -> None:
        """Reject parameter combinations that would silently misbehave."""
        if self.min_posts < MIN_POSTS_FOR_GAPS:
            msg = (
                f"min_posts must be at least {MIN_POSTS_FOR_GAPS}; one post has no gaps to measure"
            )
            raise ValueError(msg)
        if not 0 < self.quiet_window_hours < HOURS_PER_DAY:
            msg = f"quiet_window_hours must be in (0, {HOURS_PER_DAY})"
            raise ValueError(msg)
        if self.burst_factor <= 0:
            msg = "burst_factor must be positive"
            raise ValueError(msg)
        if self.min_burst_size < MIN_POSTS_FOR_GAPS:
            msg = f"min_burst_size must be at least {MIN_POSTS_FOR_GAPS}"
            raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, meant to be embedded in every report."""
        return {
            "min_posts": self.min_posts,
            "quiet_window_hours": self.quiet_window_hours,
            "burst_factor": self.burst_factor,
            "burst_ceiling_seconds": self.burst_ceiling.total_seconds(),
            "min_burst_size": self.min_burst_size,
            "include_reposts": self.include_reposts,
            "min_resolution_seconds": self.min_resolution_seconds,
        }


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    """One account's posting rhythm, computed once and reused by the features.

    Attributes:
        account_id: The account described.
        n_posts: Posts that entered the analysis.
        hour_histogram: Posts per hour of the day, in recorded time.
        quiet_share: Share of posts in the quietest contiguous window.
        quiet_start_hour: Where that window starts, in recorded time. Only
            comparable between accounts of the same corpus, never a timezone.
        gaps: Consecutive inter-post gaps in seconds.
        bursts: Index ranges of detected bursts.
        span_days: Days between first and last post.
        active_days: Distinct calendar days with at least one post.
        resolution_seconds: Apparent timestamp granularity of the source.
    """

    account_id: AccountId
    n_posts: int
    hour_histogram: tuple[int, ...]
    quiet_share: float
    quiet_start_hour: int
    gaps: tuple[float, ...]
    bursts: tuple[tuple[int, int], ...]
    span_days: float
    active_days: int
    resolution_seconds: float

    @property
    def circadian_entropy(self) -> float:
        """Normalised entropy of the hour-of-day histogram."""
        return shannon_entropy(self.hour_histogram, bins=HOURS_PER_DAY)

    @property
    def interarrival_entropy(self) -> float:
        """Normalised entropy of the log-binned gaps."""
        return interarrival_entropy(self.gaps)

    @property
    def burst_share(self) -> float:
        """Share of posts that arrived inside a burst."""
        if not self.n_posts:
            return 0.0
        inside = sum(end - start for start, end in self.bursts)
        return inside / self.n_posts

    @property
    def max_burst_size(self) -> int:
        """Posts in the largest burst, zero when there is none."""
        return max((end - start for start, end in self.bursts), default=0)

    @property
    def active_days_ratio(self) -> float:
        """Share of the observed span on which the account posted at all."""
        days = max(math.ceil(self.span_days), 1)
        return min(self.active_days / days, 1.0)

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the report layer."""
        return {
            "account_id": self.account_id,
            "n_posts": self.n_posts,
            "hour_histogram": list(self.hour_histogram),
            "quiet_share": round(self.quiet_share, 4),
            "quiet_start_hour": self.quiet_start_hour,
            "median_gap_seconds": round(median(self.gaps), 1) if self.gaps else None,
            "n_bursts": len(self.bursts),
            "max_burst_size": self.max_burst_size,
            "span_days": round(self.span_days, 2),
            "active_days": self.active_days,
            "resolution_seconds": self.resolution_seconds,
        }


def _resolution_seconds(times: Sequence[datetime]) -> float:
    """Guess the timestamp granularity of a source from the values themselves.

    An archive that truncates to the day leaves every timestamp at midnight; one
    that truncates to the minute leaves every second at zero. Detecting this is
    what keeps the inter-arrival features from reporting a confident number for
    data that cannot support one.
    """
    if not times:
        return 1.0
    if all(t.second == 0 and t.microsecond == 0 for t in times):
        if all(t.minute == 0 for t in times):
            return SECONDS_PER_DAY if all(t.hour == 0 for t in times) else 3600.0
        return 60.0
    return 1.0


def profile_accounts(
    corpus: Corpus, config: TemporalConfig | None = None
) -> dict[AccountId, TemporalProfile]:
    """Build a :class:`TemporalProfile` for every account with posts.

    Profiles are built for every account regardless of ``min_posts``: the
    threshold decides what may be *reported*, not what may be computed, and a
    report card for a small account should still be able to show its histogram.
    """
    config = config or TemporalConfig()
    profiles: dict[AccountId, TemporalProfile] = {}
    for account_id, posts in corpus.posts_by_account.items():
        eligible: list[Post] = [
            post for post in posts if config.include_reposts or not post.is_repost
        ]
        if not eligible:
            continue
        times = [post.created_at for post in eligible]
        share, start_hour = quiet_hours_share(times, window_hours=config.quiet_window_hours)
        gaps = gaps_seconds(times)
        resolution = _resolution_seconds(times)
        bursts = (
            find_bursts(
                times,
                factor=config.burst_factor,
                ceiling=config.burst_ceiling,
                min_size=config.min_burst_size,
            )
            if resolution <= config.min_resolution_seconds
            else []
        )
        profiles[account_id] = TemporalProfile(
            account_id=account_id,
            n_posts=len(times),
            hour_histogram=tuple(hour_histogram(times)),
            quiet_share=share,
            quiet_start_hour=start_hour,
            gaps=tuple(gaps),
            bursts=tuple(bursts),
            span_days=(times[-1] - times[0]).total_seconds() / SECONDS_PER_DAY,
            active_days=len({t.date() for t in times}),
            resolution_seconds=resolution,
        )
    return profiles


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="temp_circadian_entropy",
        description="Evenness of posting across the 24 hours of the day.",
        rationale=(
            "People are awake on a schedule and their posting inherits the shape of it. "
            "A process that runs continuously spreads evenly across all 24 hours, which "
            "pushes this towards its maximum. The measure is invariant to timezone."
        ),
        limitation=(
            "Shift workers, insomniacs and accounts shared across several timezones all "
            "flatten the same way, and any account with few posts looks flat by accident."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="bits_normalised",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="temp_quiet_hours_share",
        description="Share of posts inside the account's own quietest six-hour window.",
        rationale=(
            "Everyone sleeps somewhere, so a human timeline has a stretch that is nearly "
            "empty wherever their night happens to fall. Finding the quietest window per "
            "account rather than assuming a local night keeps this usable without knowing "
            "anyone's timezone."
        ),
        limitation=(
            "An account operated in shifts by several people, or one that mostly reshares "
            "automatically while its owner sleeps, has no quiet window and is not covert."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 0.25),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="temp_interarrival_entropy",
        description="Entropy of the log-binned gaps between consecutive posts.",
        rationale=(
            "Human attention is lumpy: minutes of replies, then hours of nothing, spread "
            "over many orders of magnitude. A fixed cadence collapses every gap into one "
            "bucket, and the entropy falls towards zero."
        ),
        limitation=(
            "Any scheduling tool produces the same collapse, and an account that posts "
            "only a few times leaves too few gaps to estimate a distribution from."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="bits_normalised",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=False,
    ),
    FeatureSpec(
        name="temp_median_interarrival_seconds",
        description="Median time between consecutive posts.",
        rationale=(
            "Sustained short gaps put an account outside what a person maintains by hand "
            "over a long collection window, especially combined with a flat circadian "
            "shape and an absent quiet window."
        ),
        limitation=(
            "Confounded with volume and with topic: live-posting an election night or a "
            "football match produces minutes-long medians from an entirely human account."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="seconds",
        value_range=(0.0, None),
        higher_is_more_anomalous=False,
    ),
    FeatureSpec(
        name="temp_burst_share",
        description="Share of posts arriving inside a burst, relative to the account's baseline.",
        rationale=(
            "Queued or triggered publishing tends to discharge in clumps far faster than "
            "the account's own median rhythm, rather than arriving at the irregular pace "
            "of someone typing."
        ),
        limitation=(
            "Thread writing, live commentary and catching up after a flight all look like "
            "bursts, and the threshold is relative, so a quiet account bursts more easily."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="temp_max_burst_size",
        description="Number of posts in the account's largest burst.",
        rationale=(
            "Distinguishes an account that occasionally posts twice in a row from one "
            "that empties a queue of dozens of items in a few minutes."
        ),
        limitation=(
            "A long thread published in one sitting is a single large burst and is "
            "completely ordinary human behaviour on most platforms."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="count",
        value_range=(0.0, None),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="temp_active_days_ratio",
        description="Share of days in the observed span on which the account posted.",
        rationale=(
            "People take days off: they travel, get busy, lose interest for a week. A "
            "process that never misses a day over a long span is behaving unlike its "
            "audience even when its daily volume is modest."
        ),
        limitation=(
            "Professional accounts -- newsrooms, institutions, anyone whose job is to "
            "post -- reach 1.0 legitimately, and a short collection window reaches it by "
            "accident."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
)

_SUB_DAILY_FEATURES = (
    "temp_interarrival_entropy",
    "temp_median_interarrival_seconds",
    "temp_burst_share",
    "temp_max_burst_size",
)
"""Features that a coarse-resolution source cannot support."""


class TemporalExtractor:
    """Account-level posting rhythm, as a feature frame.

    Three different kinds of silence are kept apart, because collapsing them
    would feed the ensemble a fact that is not true:

    * An account below ``min_posts`` gets ``NaN`` everywhere. There is not
      enough evidence to describe its rhythm at all.
    * An account from a day-resolution source gets its circadian and
      active-days features but ``NaN`` for everything sub-daily, because the
      timestamps cannot support those.
    * An account with enough posts and no bursts gets ``0``. That is a
      measurement, not a gap.
    """

    name = "temporal"
    specs = SPECS

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()

    def extract(self, corpus: Corpus) -> pd.DataFrame:
        """Compute the temporal features for every account in ``corpus``."""
        frame = empty_frame(corpus, self.specs)
        for account_id, profile in profile_accounts(corpus, self.config).items():
            if account_id not in frame.index or profile.n_posts < self.config.min_posts:
                continue
            frame.loc[account_id, "temp_circadian_entropy"] = profile.circadian_entropy
            frame.loc[account_id, "temp_quiet_hours_share"] = profile.quiet_share
            frame.loc[account_id, "temp_active_days_ratio"] = profile.active_days_ratio
            if profile.resolution_seconds > self.config.min_resolution_seconds:
                continue
            frame.loc[account_id, "temp_interarrival_entropy"] = profile.interarrival_entropy
            frame.loc[account_id, "temp_median_interarrival_seconds"] = (
                median(profile.gaps) if profile.gaps else float("nan")
            )
            frame.loc[account_id, "temp_burst_share"] = profile.burst_share
            frame.loc[account_id, "temp_max_burst_size"] = float(profile.max_burst_size)
        return frame
