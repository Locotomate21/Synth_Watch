"""Tests for the posting-rhythm features.

The load-bearing property is rotation invariance: none of these features may
change when the whole timeline moves to another timezone, because the corpus
never says which timezone an account is in.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from synthwatch.detect.temporal import (
    HOURS_PER_DAY,
    TemporalConfig,
    TemporalExtractor,
    circadian_entropy,
    find_bursts,
    gaps_seconds,
    hour_histogram,
    interarrival_entropy,
    profile_accounts,
    quiet_hours_share,
    shannon_entropy,
)
from synthwatch.models import build_corpus
from tests.conftest import make_account, make_corpus
from tests.detect.fixtures_temporal import (
    ANCHOR,
    bursty_posts,
    coarse_posts,
    human_posts,
    metronome_posts,
    repost_heavy_posts,
    rhythm_corpus,
    shifted,
    sparse_posts,
)


def at(*hours: float) -> list[datetime]:
    """Timestamps at the given hours after the anchor."""
    return [ANCHOR + timedelta(hours=hour) for hour in hours]


# ------------------------------------------------------------------ primitives


class TestEntropy:
    def test_everything_in_one_bin_is_zero(self):
        assert shannon_entropy([10, 0, 0, 0]) == 0.0

    def test_a_flat_histogram_is_one(self):
        assert shannon_entropy([5, 5, 5, 5]) == pytest.approx(1.0)

    def test_empty_input_is_zero_not_an_error(self):
        assert shannon_entropy([]) == 0.0
        assert shannon_entropy([0, 0]) == 0.0

    def test_entropy_is_never_negative_zero(self):
        # -0.0 renders as "-0.000" in a report, which reads like a bug.
        value = shannon_entropy([7])
        assert value == 0.0
        assert not str(value).startswith("-")

    def test_bins_argument_keeps_sparse_histograms_comparable(self):
        # Two gaps in two buckets out of 21 possible is not maximal entropy.
        assert shannon_entropy([1, 1], bins=21) < 0.5
        assert shannon_entropy([1, 1]) == pytest.approx(1.0)


class TestCircadian:
    def test_a_single_hour_is_zero(self):
        assert circadian_entropy(at(3, 27, 51)) == 0.0

    def test_every_hour_equally_is_one(self):
        assert circadian_entropy(at(*range(HOURS_PER_DAY))) == pytest.approx(1.0)

    def test_rotation_leaves_it_unchanged(self):
        times = at(8, 9, 14, 20, 21)
        rotated = [moment + timedelta(hours=13) for moment in times]
        assert circadian_entropy(times) == pytest.approx(circadian_entropy(rotated))

    def test_histogram_bins_by_hour(self):
        counts = hour_histogram(at(0, 0, 5, 23))
        assert counts[0] == 2
        assert counts[5] == 1
        assert counts[23] == 1


class TestQuietHours:
    def test_a_sleeper_has_an_empty_window(self):
        share, _ = quiet_hours_share(at(9, 11, 13, 15, 17, 19, 21))
        assert share == 0.0

    def test_round_the_clock_posting_fills_it(self):
        share, _ = quiet_hours_share(at(*range(HOURS_PER_DAY)))
        assert share == pytest.approx(6 / 24)

    def test_the_window_wraps_around_midnight(self):
        # Quiet from 22:00 to 04:00: the window has to cross midnight to find it.
        share, start = quiet_hours_share(at(4, 6, 10, 14, 18, 21))
        assert share == 0.0
        assert start == 22

    def test_no_posts_is_zero_not_an_error(self):
        assert quiet_hours_share([]) == (0.0, 0)


class TestGapsAndBursts:
    def test_gaps_are_consecutive_differences(self):
        assert gaps_seconds(at(0, 1, 3)) == [3600.0, 7200.0]

    def test_one_post_has_no_gaps(self):
        assert gaps_seconds(at(0)) == []

    def test_a_metronome_has_near_zero_entropy(self):
        assert interarrival_entropy([2700.0] * 50) == 0.0

    def test_spread_gaps_raise_the_entropy(self):
        assert interarrival_entropy([10.0, 300.0, 9000.0, 200000.0]) > 0.3

    def test_a_clump_is_found(self):
        times = at(0, 0.001, 0.002, 0.003, 40, 80, 120)
        assert find_bursts(times, min_size=3) == [(0, 4)]

    def test_runs_below_the_minimum_are_not_bursts(self):
        assert find_bursts(at(0, 0.001, 40, 80), min_size=3) == []

    def test_an_even_cadence_never_bursts(self):
        assert find_bursts(at(*[n * 0.75 for n in range(40)])) == []

    def test_the_ceiling_stops_a_slow_account_from_bursting(self):
        # Three posts in six hours, for an account that posts twice a month:
        # fast relative to its own mean, but not a burst in any useful sense.
        times = at(0, 3, 6, 720, 1440)
        assert find_bursts(times, ceiling=timedelta(hours=1)) == []
        assert find_bursts(times, ceiling=timedelta(hours=12)) == [(0, 3)]

    def test_the_baseline_is_the_mean_not_the_median(self):
        # Most posts are inside bursts here, so the median gap *is* the
        # intra-burst gap and a median baseline would find nothing.
        times = at(*[0, 0.01, 0.02, 0.03, 48, 48.01, 48.02, 48.03])
        assert find_bursts(times, min_size=3) == [(0, 4), (4, 8)]


# -------------------------------------------------------------------- profiles


class TestProfiles:
    def test_every_posting_account_gets_a_profile(self):
        profiles = profile_accounts(rhythm_corpus())
        assert set(profiles) == {"human", "metro", "bursty", "sparse"}

    def test_a_small_account_still_gets_a_profile(self):
        # min_posts gates what may be *reported*, not what may be computed: a
        # report card for a quiet account should still show its histogram.
        profiles = profile_accounts(make_corpus(sparse_posts()))
        assert profiles["sparse"].n_posts == 5

    def test_the_human_sleeps_and_takes_days_off(self):
        profile = profile_accounts(make_corpus(human_posts()))["human"]
        assert profile.quiet_share == 0.0
        assert profile.circadian_entropy < 0.9
        assert profile.active_days_ratio < 1.0
        assert profile.interarrival_entropy > 0.5

    def test_the_metronome_has_no_night_and_no_variation(self):
        profile = profile_accounts(make_corpus(metronome_posts()))["metro"]
        assert profile.circadian_entropy > 0.95
        assert profile.quiet_share == pytest.approx(0.25, abs=0.01)
        assert profile.interarrival_entropy == 0.0
        assert profile.active_days_ratio == 1.0
        assert profile.burst_share == 0.0

    def test_the_bursty_account_is_almost_all_burst(self):
        profile = profile_accounts(make_corpus(bursty_posts()))["bursty"]
        assert profile.burst_share == 1.0
        assert profile.max_burst_size == 6
        assert profile.interarrival_entropy < 0.3

    def test_coarse_timestamps_are_detected(self):
        profile = profile_accounts(make_corpus(coarse_posts()))["coarse"]
        assert profile.resolution_seconds == 86_400.0
        assert profile.bursts == ()

    def test_minute_resolution_is_detected(self):
        profile = profile_accounts(make_corpus(metronome_posts()))["metro"]
        assert profile.resolution_seconds == 60.0

    def test_profile_is_serialisable(self):
        exported = profile_accounts(make_corpus(metronome_posts()))["metro"].as_dict()
        assert exported["n_posts"] == 320
        assert exported["median_gap_seconds"] == 2700.0
        assert len(cast("list[int]", exported["hour_histogram"])) == HOURS_PER_DAY


class TestReposts:
    def test_reshares_count_by_default(self):
        profile = profile_accounts(make_corpus(repost_heavy_posts()))["amplifier"]
        assert profile.n_posts == 210
        assert profile.quiet_share > 0.0

    def test_excluding_them_makes_the_same_account_look_asleep(self):
        # Both readings are correct. Which one is reported is a choice, and the
        # config records it.
        config = TemporalConfig(include_reposts=False)
        profile = profile_accounts(make_corpus(repost_heavy_posts()), config)["amplifier"]
        assert profile.n_posts == 120
        assert profile.quiet_share == 0.0


# ------------------------------------------------------------- the key property


class TestRotationInvariance:
    @pytest.mark.parametrize("hours", [3, 7, 12, -5, 13])
    def test_features_survive_a_timezone_shift(self, hours: float):
        original = profile_accounts(make_corpus(human_posts()))["human"]
        rotated = profile_accounts(make_corpus(shifted(human_posts(), hours)))["human"]
        assert rotated.circadian_entropy == pytest.approx(original.circadian_entropy)
        assert rotated.quiet_share == pytest.approx(original.quiet_share)
        assert rotated.interarrival_entropy == pytest.approx(original.interarrival_entropy)
        assert rotated.burst_share == pytest.approx(original.burst_share)
        assert rotated.max_burst_size == original.max_burst_size

    def test_the_quiet_window_moves_with_the_clock(self):
        # The only thing that may change is *where* the quiet window sits, and
        # it must move exactly as far as the timeline did. The histogram needs a
        # unique minimum for the question to have one answer.
        times: list[datetime] = []
        for hour in range(HOURS_PER_DAY):
            repeats = 1 if 2 <= hour <= 7 else 2
            times += at(*[hour + HOURS_PER_DAY * day for day in range(repeats)])

        share, start = quiet_hours_share(times)
        assert start == 2

        rotated_share, rotated_start = quiet_hours_share(
            [moment + timedelta(hours=5) for moment in times]
        )
        assert rotated_start == 7
        assert rotated_share == pytest.approx(share)

    def test_a_half_hour_timezone_is_the_documented_exception(self):
        # India and Nepal sit at :30 and :45 offsets, which move posts across
        # hour boundaries instead of rotating the histogram. The invariance is
        # then approximate rather than exact, and the drift is small.
        original = profile_accounts(make_corpus(human_posts()))["human"]
        rotated = profile_accounts(make_corpus(shifted(human_posts(), 5.5)))["human"]
        drift = abs(rotated.circadian_entropy - original.circadian_entropy)
        assert 0 < drift < 0.05
        assert rotated.quiet_share == original.quiet_share
        assert rotated.burst_share == pytest.approx(original.burst_share)


# -------------------------------------------------------------------- features


class TestExtractor:
    def test_every_feature_is_documented_and_prefixed(self):
        for spec in TemporalExtractor().specs:
            assert spec.name.startswith("temp_")
            assert len(spec.rationale) >= 40
            assert len(spec.limitation) >= 40

    def test_frame_covers_every_account(self):
        corpus = rhythm_corpus()
        frame = TemporalExtractor().extract(corpus)
        assert set(frame.index) == {a.account_id for a in corpus.accounts}
        assert list(frame.columns) == [s.name for s in TemporalExtractor().specs]

    def test_the_metronome_scores_as_expected(self):
        row = TemporalExtractor().extract(rhythm_corpus()).loc["metro"]
        assert row["temp_circadian_entropy"] > 0.95
        assert row["temp_quiet_hours_share"] == pytest.approx(0.25, abs=0.01)
        assert row["temp_interarrival_entropy"] == 0.0
        assert row["temp_median_interarrival_seconds"] == 2700.0
        assert row["temp_active_days_ratio"] == 1.0
        assert row["temp_burst_share"] == 0.0

    def test_the_human_scores_as_expected(self):
        row = TemporalExtractor().extract(rhythm_corpus()).loc["human"]
        assert row["temp_quiet_hours_share"] == 0.0
        assert row["temp_interarrival_entropy"] > 0.5
        assert row["temp_active_days_ratio"] < 1.0

    def test_the_bursty_account_scores_as_expected(self):
        row = TemporalExtractor().extract(rhythm_corpus()).loc["bursty"]
        assert row["temp_burst_share"] == 1.0
        assert row["temp_max_burst_size"] == 6

    def test_too_few_posts_stays_nan_everywhere(self):
        # "Not enough evidence" must never be recorded as "measured, normal".
        row = TemporalExtractor().extract(rhythm_corpus()).loc["sparse"]
        assert row.isna().all()

    def test_coarse_sources_keep_the_daily_features_and_drop_the_rest(self):
        row = TemporalExtractor().extract(make_corpus(coarse_posts())).loc["coarse"]
        assert not math.isnan(cast("float", row["temp_circadian_entropy"]))
        assert row["temp_active_days_ratio"] == 1.0
        for name in (
            "temp_interarrival_entropy",
            "temp_median_interarrival_seconds",
            "temp_burst_share",
            "temp_max_burst_size",
        ):
            assert math.isnan(cast("float", row[name])), f"{name} must be NaN"

    def test_measured_absence_of_bursts_is_zero_not_nan(self):
        row = TemporalExtractor().extract(rhythm_corpus()).loc["metro"]
        assert row["temp_burst_share"] == 0.0

    def test_silent_accounts_stay_nan(self):
        corpus = build_corpus([make_account("quiet")], metronome_posts())
        assert TemporalExtractor().extract(corpus).loc["quiet"].isna().all()


class TestConfig:
    def test_min_posts_below_two_is_refused(self):
        with pytest.raises(ValueError, match="min_posts"):
            TemporalConfig(min_posts=1)

    def test_quiet_window_must_fit_in_a_day(self):
        with pytest.raises(ValueError, match="quiet_window_hours"):
            TemporalConfig(quiet_window_hours=24)

    def test_burst_factor_must_be_positive(self):
        with pytest.raises(ValueError, match="burst_factor"):
            TemporalConfig(burst_factor=0.0)

    def test_min_burst_size_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="min_burst_size"):
            TemporalConfig(min_burst_size=1)

    def test_config_is_serialisable_for_the_report(self):
        exported = TemporalConfig().as_dict()
        assert exported["min_posts"] == 20
        assert exported["burst_ceiling_seconds"] == 3600.0


def test_utc_anchor_is_what_the_fixtures_assume():
    assert datetime(2024, 3, 1, tzinfo=UTC) == ANCHOR
