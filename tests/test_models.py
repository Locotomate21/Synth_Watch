"""Tests for the internal schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from synthwatch.models import Account, Corpus, LabelRecord, Post, build_corpus
from synthwatch.types import Label, LabelMethod, Platform, PostKind
from tests.conftest import EPOCH, make_account, make_corpus, make_post


class TestTimestamps:
    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValidationError, match="naive datetime"):
            Post(
                post_id="p1",
                account_id="a1",
                platform=Platform.GENERIC,
                created_at=datetime(2024, 3, 1, 12, 0),  # noqa: DTZ001 - the point of the test
            )

    def test_aware_datetime_is_converted_to_utc(self):
        madrid = timezone(timedelta(hours=2))
        post = make_post("p1", "a1")
        shifted = post.model_copy(update={"created_at": datetime(2024, 3, 1, 14, 0, tzinfo=madrid)})
        # model_copy skips validation, so validate explicitly through the ctor.
        revalidated = Post.model_validate(shifted.model_dump())
        assert revalidated.created_at.tzinfo is UTC
        assert revalidated.created_at == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    def test_account_age_uses_the_reference_instant_not_now(self):
        account = make_account("a1", created_days_ago=100)
        assert account.age_days(EPOCH) == pytest.approx(100.0)
        assert account.age_days(EPOCH + timedelta(days=365)) == pytest.approx(465.0)

    def test_age_is_none_without_a_creation_date(self):
        assert Account(account_id="a1", platform=Platform.GENERIC).age_days(EPOCH) is None


class TestRecords:
    def test_records_are_frozen(self):
        post = make_post("p1", "a1")
        with pytest.raises(ValidationError):
            post.text = "edited"  # type: ignore[misc]

    def test_unknown_fields_are_refused_not_swallowed(self):
        with pytest.raises(ValidationError):
            Account(account_id="a1", platform=Platform.GENERIC, follower_count=3)  # type: ignore[call-arg]

    def test_unmapped_payload_survives_in_extra(self):
        account = make_account("a1", extra={"karma": 42})
        assert account.extra["karma"] == 42

    def test_handle_is_stripped_of_at_sign(self):
        assert make_account("a1", handle="  @Nombre ").handle == "Nombre"

    def test_key_is_platform_scoped(self):
        one = make_account("shared", platform=Platform.REDDIT)
        two = make_account("shared", platform=Platform.MASTODON)
        assert len({one.key, two.key}) == 2

    def test_negative_counts_are_rejected(self):
        with pytest.raises(ValidationError):
            make_account("a1", followers=-1)

    def test_text_is_stored_verbatim(self):
        raw = "  DOS  espacios\ty tab\n"
        assert make_post("p1", "a1", text=raw).text == raw


class TestCorpus:
    def test_posts_are_sorted_chronologically(self):
        corpus = make_corpus(
            [
                make_post("p3", "a1", offset_minutes=30),
                make_post("p1", "a1", offset_minutes=0),
                make_post("p2", "a1", offset_minutes=10),
            ]
        )
        assert [p.post_id for p in corpus.posts] == ["p1", "p2", "p3"]

    def test_ties_break_on_post_id_for_determinism(self):
        corpus = make_corpus(
            [make_post("pb", "a1", offset_minutes=0), make_post("pa", "a2", offset_minutes=0)]
        )
        assert [p.post_id for p in corpus.posts] == ["pa", "pb"]

    def test_posts_by_account_is_chronological(self, quiet_corpus: Corpus):
        posts = quiet_corpus.posts_by_account["a1"]
        assert [p.post_id for p in posts] == ["p1", "p4"]

    def test_orphan_posts_are_reported_not_raised(self):
        corpus = Corpus(accounts=(make_account("a1"),), posts=(make_post("p1", "ghost"),))
        assert corpus.orphan_post_ids == ("p1",)

    def test_strict_build_refuses_orphans(self):
        with pytest.raises(ValueError, match="unknown accounts"):
            build_corpus([make_account("a1")], [make_post("p1", "ghost")], strict=True)

    def test_silent_accounts_are_reported(self):
        corpus = Corpus(
            accounts=(make_account("a1"), make_account("a2")),
            posts=(make_post("p1", "a1"),),
        )
        assert corpus.silent_account_ids == ("a2",)

    def test_time_span_of_empty_corpus_is_none(self):
        assert Corpus().time_span is None

    def test_filter_posts_keeps_accounts(self, quiet_corpus: Corpus):
        filtered = quiet_corpus.filter_posts(lambda p: p.account_id == "a1")
        assert len(filtered) == 2
        assert len(filtered.accounts) == 3
        assert filtered.silent_account_ids == ("a2", "a3")

    def test_subset_drops_accounts_and_posts(self, quiet_corpus: Corpus):
        subset = quiet_corpus.subset(["a1"])
        assert {a.account_id for a in subset.accounts} == {"a1"}
        assert len(subset) == 2

    def test_merge_deduplicates_by_key(self, quiet_corpus: Corpus):
        merged = quiet_corpus.merge(quiet_corpus)
        assert len(merged) == len(quiet_corpus)
        assert len(merged.accounts) == len(quiet_corpus.accounts)

    def test_summary_reports_integrity_counters(self, quiet_corpus: Corpus):
        summary = quiet_corpus.summary()
        assert summary["n_posts"] == 4
        assert summary["n_accounts"] == 3
        assert summary["orphan_posts"] == 0
        assert summary["platforms"] == ["generic"]


class TestLabels:
    def test_label_carries_provenance(self):
        record = LabelRecord(
            account_id="a1",
            platform=Platform.TWITTER,
            label=Label.INFO_OPERATION,
            dataset="twitter-io-archive/2019-06",
            method=LabelMethod.PLATFORM_ENFORCEMENT,
        )
        assert record.dataset
        assert record.method is LabelMethod.PLATFORM_ENFORCEMENT

    def test_no_verdict_field_exists_on_account(self):
        # Guards the ethical contract: predictions never get written back onto
        # a record, so a serialised corpus can never look like an accusation.
        forbidden = {"is_bot", "bot_score", "label", "verdict", "suspicious"}
        assert forbidden.isdisjoint(Account.model_fields)
        assert forbidden.isdisjoint(Post.model_fields)


def test_post_kind_helpers():
    assert make_post("p1", "a1", kind=PostKind.REPLY).is_reply
    assert make_post("p2", "a1", kind=PostKind.REPOST).is_repost
    assert not make_post("p3", "a1").is_reply
