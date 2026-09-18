"""Tests for the profile-shaped features.

The recurring theme is that absent metadata must stay absent. These features
are cheap to compute and easy to fake into existence out of a gap in the data,
which is the failure this suite is built to catch.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from synthwatch.detect.account import (
    PROFILE_FIELDS,
    AccountConfig,
    AccountExtractor,
    available_profile_fields,
    digit_ratio,
    handle_shape,
    profile_completeness,
    trailing_digits,
)
from synthwatch.detect.stats import character_entropy, mean, safe_ratio
from synthwatch.models import Account, build_corpus
from synthwatch.types import Platform
from tests.conftest import EPOCH, make_account, make_post

LATER = EPOCH + timedelta(days=365)


# ------------------------------------------------------------------ primitives


class TestHandleShape:
    @pytest.mark.parametrize(
        ("handle", "expected"),
        [("maria", 0.0), ("maria1990", 4 / 9), ("12345678", 1.0), ("", 0.0)],
    )
    def test_digit_ratio(self, handle: str, expected: float):
        assert digit_ratio(handle) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("handle", "expected"),
        [("maria", 0), ("maria1990", 4), ("m4ria", 0), ("user48210573", 8)],
    )
    def test_trailing_digits(self, handle: str, expected: int):
        assert trailing_digits(handle) == expected

    def test_digits_inside_the_handle_do_not_count_as_trailing(self):
        # A birth year in the middle is a different shape from a generated suffix.
        assert trailing_digits("m4ria_lopez") == 0
        assert digit_ratio("m4ria_lopez") > 0

    def test_a_repetitive_handle_has_low_entropy(self):
        assert character_entropy("aaaaaaaa") == 0.0
        assert character_entropy("xqjvkbzm") == pytest.approx(1.0)

    def test_shape_bundles_the_measures(self):
        shape = handle_shape("Maria1990")
        assert shape.length == 9
        assert shape.trailing_digits == 4
        assert shape.entropy > 0


class TestCompleteness:
    def test_an_empty_profile_scores_zero(self):
        account = Account(account_id="a1", platform=Platform.GENERIC)
        assert profile_completeness(account) == 0.0

    def test_a_full_profile_scores_one(self):
        account = Account(
            account_id="a1",
            platform=Platform.GENERIC,
            display_name="Maria",
            description="Vecina del barrio",
            location="Madrid",
            url="https://ejemplo.org",
        )
        assert profile_completeness(account) == 1.0

    def test_a_default_avatar_counts_against_it_when_recorded(self):
        def account(**extra: object) -> Account:
            return Account(
                account_id="a1", platform=Platform.GENERIC, display_name="Maria", **extra
            )

        neutral = profile_completeness(account())
        default_avatar = profile_completeness(account(has_default_profile_image=True))
        own_avatar = profile_completeness(account(has_default_profile_image=False))
        assert neutral is not None
        assert default_avatar is not None
        assert own_avatar is not None
        assert default_avatar < neutral < own_avatar

    def test_nothing_to_count_is_none_not_zero(self):
        account = Account(account_id="a1", platform=Platform.GENERIC)
        assert profile_completeness(account, ()) is None

    def test_the_counted_fields_are_configurable(self):
        account = Account(account_id="a1", platform=Platform.GENERIC, location="Madrid")
        assert profile_completeness(account, ("location",)) == 1.0
        partial = profile_completeness(account, PROFILE_FIELDS)
        assert partial is not None
        assert partial < 1.0


class TestSafeRatio:
    def test_a_zero_denominator_is_unknown_not_infinite(self):
        assert safe_ratio(5, 0) is None

    def test_missing_inputs_propagate(self):
        assert safe_ratio(None, 10) is None
        assert safe_ratio(10, None) is None

    def test_an_ordinary_division(self):
        assert safe_ratio(3, 4) == 0.75

    def test_mean_of_nothing_is_none(self):
        assert mean([]) is None
        assert mean([1.0, 3.0]) == 2.0


# -------------------------------------------------------------------- reference


class TestReferenceInstant:
    def test_an_explicit_reference_wins(self):
        config = AccountConfig(reference=LATER)
        corpus = build_corpus([], [], collected_at=EPOCH)
        assert config.reference_for(corpus) == LATER

    def test_otherwise_the_corpus_collection_time(self):
        corpus = build_corpus([], [], collected_at=EPOCH)
        assert AccountConfig().reference_for(corpus) == EPOCH

    def test_then_the_latest_post_collection_time(self):
        corpus = build_corpus([], [make_post("p1", "a1")])
        assert AccountConfig().reference_for(corpus) == EPOCH

    def test_an_empty_corpus_has_no_reference(self):
        assert AccountConfig().reference_for(build_corpus([], [])) is None

    def test_age_does_not_drift_when_the_analysis_is_re_run_later(self):
        # The whole point of not using "now": the same corpus must produce the
        # same numbers next year.
        account = make_account("a1", created_days_ago=100)
        corpus = build_corpus([account], [make_post("p1", "a1")], collected_at=EPOCH)
        first = AccountExtractor().extract(corpus)
        again = AccountExtractor(AccountConfig(reference=EPOCH)).extract(corpus)
        assert first.loc["a1", "acct_age_days"] == pytest.approx(100.0)
        assert again.loc["a1", "acct_age_days"] == pytest.approx(100.0)


# --------------------------------------------------------------------- features


class TestExtractor:
    def test_every_feature_is_documented_and_prefixed(self):
        for spec in AccountExtractor().specs:
            assert spec.name.startswith("acct_")
            assert len(spec.rationale) >= 40
            assert len(spec.limitation) >= 40

    def test_a_well_described_account_scores_on_everything(self):
        account = make_account(
            "a1",
            created_days_ago=400,
            followers=300,
            following=100,
            handle="maria_lopez",
            description="Vecina del barrio",
            location="Madrid",
            url="https://ejemplo.org",
            post_count=800,
        )
        posts = [make_post("p1", "a1", offset_minutes=0)]
        frame = AccountExtractor().extract(build_corpus([account], posts, collected_at=EPOCH))
        row = frame.loc["a1"]
        assert row["acct_age_days"] == pytest.approx(400.0)
        assert row["acct_followback_ratio"] == pytest.approx(0.75)
        assert row["acct_profile_completeness"] == 1.0
        assert row["acct_posts_per_day"] == pytest.approx(2.0)
        assert row["acct_handle_digit_ratio"] == 0.0
        assert row["acct_handle_trailing_digits"] == 0.0
        assert row["acct_dormancy_days"] == pytest.approx(400.0)
        assert not row.isna().any()

    def test_a_generated_looking_account_scores_the_other_way(self):
        # Paired with a described account, so the corpus clearly exposes the
        # profile fields this one left empty.
        described = make_account("human", description="Vecina del barrio", location="Madrid")
        account = make_account(
            "bot",
            created_days_ago=12,
            followers=3,
            following=900,
            handle="user48210573",
            display_name=None,
            post_count=2400,
        )
        frame = AccountExtractor().extract(
            build_corpus(
                [described, account],
                [make_post("p1", "bot"), make_post("p2", "human")],
                collected_at=EPOCH,
            )
        )
        row = frame.loc["bot"]
        assert row["acct_age_days"] == pytest.approx(12.0)
        assert row["acct_handle_trailing_digits"] == 8.0
        assert row["acct_handle_digit_ratio"] > 0.6
        assert row["acct_followback_ratio"] < 0.01
        assert row["acct_posts_per_day"] > 100
        assert row["acct_profile_completeness"] < 0.5

    def test_dormancy_measures_the_gap_before_the_first_observed_post(self):
        account = make_account("a1", created_days_ago=900)
        posts = [make_post("p1", "a1", offset_minutes=0)]
        frame = AccountExtractor().extract(build_corpus([account], posts, collected_at=EPOCH))
        assert frame.loc["a1", "acct_dormancy_days"] == pytest.approx(900.0)

    def test_dormancy_never_goes_negative(self):
        # A post older than the recorded creation date happens in real exports.
        account = make_account("a1", created_days_ago=1)
        posts = [make_post("p1", "a1", offset_minutes=-60 * 24 * 10)]
        frame = AccountExtractor().extract(build_corpus([account], posts, collected_at=EPOCH))
        assert frame.loc["a1", "acct_dormancy_days"] == 0.0


class TestMissingMetadata:
    def test_an_account_with_no_metadata_is_nan_everywhere(self):
        account = Account(account_id="bare", platform=Platform.GENERIC)
        frame = AccountExtractor().extract(
            build_corpus([account], [make_post("p1", "bare")], collected_at=EPOCH)
        )
        assert frame.loc["bare"].isna().all()

    def test_a_missing_handle_leaves_only_the_handle_features_nan(self):
        account = make_account("a1", handle=None)
        frame = AccountExtractor().extract(
            build_corpus([account], [make_post("p1", "a1")], collected_at=EPOCH)
        )
        row = frame.loc["a1"]
        assert math.isnan(cast("float", row["acct_handle_digit_ratio"]))
        assert math.isnan(cast("float", row["acct_handle_entropy"]))
        assert not math.isnan(cast("float", row["acct_age_days"]))

    def test_a_short_handle_has_no_measurable_shape(self):
        frame = AccountExtractor().extract(
            build_corpus(
                [make_account("a1", handle="jp")], [make_post("p1", "a1")], collected_at=EPOCH
            )
        )
        assert math.isnan(cast("float", frame.loc["a1", "acct_handle_entropy"]))

    def test_missing_follower_counts_stay_nan_rather_than_becoming_zero(self):
        # Reading "not exposed by this platform" as "nobody follows them" would
        # invent the strongest possible value for the feature.
        account = make_account("a1", followers=None, following=None)
        frame = AccountExtractor().extract(
            build_corpus([account], [make_post("p1", "a1")], collected_at=EPOCH)
        )
        assert math.isnan(cast("float", frame.loc["a1", "acct_followback_ratio"]))

    def test_an_account_with_no_followers_at_all_is_zero_not_nan(self):
        account = make_account("a1", followers=0, following=50)
        frame = AccountExtractor().extract(
            build_corpus([account], [make_post("p1", "a1")], collected_at=EPOCH)
        )
        assert frame.loc["a1", "acct_followback_ratio"] == 0.0

    def test_a_missing_creation_date_drops_age_dormancy_and_rate(self):
        account = Account(
            account_id="a1", platform=Platform.GENERIC, handle="maria_lopez", post_count=500
        )
        frame = AccountExtractor().extract(
            build_corpus([account], [make_post("p1", "a1")], collected_at=EPOCH)
        )
        row = frame.loc["a1"]
        for name in ("acct_age_days", "acct_posts_per_day", "acct_dormancy_days"):
            assert math.isnan(cast("float", row[name])), name
        assert not math.isnan(cast("float", row["acct_handle_entropy"]))

    def test_an_empty_profile_scores_zero_when_the_source_exposes_the_field(self):
        # One account fills in a bio, so the source clearly carries bios; the
        # other one left it blank, and that is a measurement of 0, not a gap.
        described = make_account("described", description="Vecina del barrio")
        blank = make_account("blank", display_name=None, description=None)
        frame = AccountExtractor().extract(
            build_corpus([described, blank], [make_post("p1", "described")], collected_at=EPOCH)
        )
        assert frame.loc["blank", "acct_profile_completeness"] == 0.0
        assert cast("float", frame.loc["described", "acct_profile_completeness"]) > 0.0

    def test_a_source_with_no_profile_metadata_at_all_stays_nan(self):
        # Every account bare: that is an export without profile fields, and
        # scoring everyone 0 would invent a corpus-wide signal out of it.
        accounts = [
            Account(account_id="a1", platform=Platform.GENERIC),
            Account(account_id="a2", platform=Platform.GENERIC),
        ]
        frame = AccountExtractor().extract(
            build_corpus(accounts, [make_post("p1", "a1")], collected_at=EPOCH)
        )
        assert math.isnan(cast("float", frame.loc["a1", "acct_profile_completeness"]))

    def test_availability_is_decided_per_corpus(self):
        bare = build_corpus([make_account("a1", display_name=None)], [], collected_at=EPOCH)
        rich = build_corpus([make_account("a1", location="Madrid")], [], collected_at=EPOCH)
        assert "location" not in available_profile_fields(bare)
        assert "location" in available_profile_fields(rich)

    def test_a_silent_account_still_gets_profile_features(self):
        # No posts in the corpus does not mean no profile to describe.
        frame = AccountExtractor().extract(
            build_corpus([make_account("quiet")], [], collected_at=EPOCH)
        )
        assert frame.loc["quiet", "acct_age_days"] == pytest.approx(900.0)
        assert math.isnan(cast("float", frame.loc["quiet", "acct_dormancy_days"]))

    def test_an_orphan_author_gets_a_row_of_nan(self):
        # A post whose author was never crawled: there is a row, and nothing in it.
        frame = AccountExtractor().extract(
            build_corpus([], [make_post("p1", "ghost")], collected_at=EPOCH)
        )
        assert frame.loc["ghost"].isna().all()


class TestConfig:
    def test_config_is_serialisable_for_the_report(self):
        exported = AccountConfig(reference=EPOCH).as_dict()
        assert exported["reference"] == EPOCH.isoformat()
        assert exported["min_handle_length"] == 4

    def test_the_frame_covers_every_account_and_column(self):
        corpus = build_corpus(
            [make_account("a1"), make_account("a2")],
            [make_post("p1", "a1")],
            collected_at=EPOCH,
        )
        frame = AccountExtractor().extract(corpus)
        assert set(frame.index) == {"a1", "a2"}
        assert list(frame.columns) == [s.name for s in AccountExtractor().specs]


def test_epoch_reference_is_what_the_fixtures_assume():
    assert datetime(2024, 3, 1, 12, 0, tzinfo=UTC) == EPOCH
