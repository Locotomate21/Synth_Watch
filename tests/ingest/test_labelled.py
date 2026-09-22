"""Tests for the labelled-dataset loaders.

The theme here is refusing to guess. An ambiguous numeric class, an
unrecognised class string and a single-class label set are all situations where
a loader could produce something plausible and wrong, and each one is pinned by
a test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from synthwatch.cli import main
from synthwatch.ingest.labelled import (
    IOArchiveAdapter,
    LabelCoverage,
    UnknownClassError,
    attach_labels,
    label_coverage,
    read_label_table,
)
from synthwatch.ingest.native import write_corpus
from synthwatch.models import build_corpus
from synthwatch.types import Label, LabelMethod, Platform, PostKind
from tests.conftest import make_account, make_post

VAROL_STYLE = "u1\tbot\nu2\thuman\nu3\tbot\n"
NUMERIC_STYLE = "u1\t1\nu2\t0\nu3\t1\n"
CSV_STYLE = "user_id,label\nu1,social_spambot_1\nu2,genuine_account\n"

IO_TWEETS = (
    "tweetid,userid,tweet_time,tweet_text,is_retweet,in_reply_to_tweetid,"
    "quoted_tweet_tweetid,hashtags,tweet_client_name,tweet_language\n"
    't1,u1,2019-01-05 10:00,Mensaje original del archivo,false,,,"[elecciones, ahora]",'
    "Twitter Web Client,es\n"
    "t2,u2,2019-01-05 10:04,Otro mensaje del archivo,true,,,[],TweetDeck,es\n"
    "t3,u1,2019-01-05 11:00,Respuesta a alguien del archivo,false,t2,,[],"
    "Twitter Web Client,es\n"
)

IO_USERS = (
    "userid,user_screen_name,user_display_name,user_profile_description,"
    "follower_count,following_count,account_creation_date\n"
    "u1,cuenta_uno,Cuenta Uno,Perfil de prueba,1200,900,2016-04-01\n"
    "u2,cuenta_dos,Cuenta Dos,,15,2400,2016-04-02\n"
)


@pytest.fixture
def varol_file(tmp_path: Path) -> Path:
    path = tmp_path / "varol-2017.dat"
    path.write_text(VAROL_STYLE, encoding="utf-8")
    return path


@pytest.fixture
def io_tweets(tmp_path: Path) -> Path:
    path = tmp_path / "iran_201906_1_tweets_csv_hashed.csv"
    path.write_text(IO_TWEETS, encoding="utf-8")
    return path


@pytest.fixture
def io_users(tmp_path: Path) -> Path:
    path = tmp_path / "iran_201906_1_users_csv_hashed.csv"
    path.write_text(IO_USERS, encoding="utf-8")
    return path


# ---------------------------------------------------------------- annotations


class TestLabelTable:
    def test_a_tab_separated_annotation_file(self, varol_file: Path):
        records = read_label_table(varol_file, dataset="indiana-bot-repository/varol-2017")
        assert len(records) == 3
        assert records[0].account_id == "u1"
        assert records[0].label is Label.AUTOMATED
        assert records[1].label is Label.ORGANIC

    def test_provenance_is_recorded_on_every_label(self, varol_file: Path):
        records = read_label_table(
            varol_file,
            dataset="indiana-bot-repository/varol-2017",
            method=LabelMethod.HUMAN_ANNOTATION,
        )
        assert all(r.dataset == "indiana-bot-repository/varol-2017" for r in records)
        assert all(r.method is LabelMethod.HUMAN_ANNOTATION for r in records)

    def test_the_source_class_is_kept_verbatim_in_the_notes(self, tmp_path: Path):
        path = tmp_path / "cresci.csv"
        path.write_text(CSV_STYLE, encoding="utf-8")
        records = read_label_table(path, dataset="indiana-bot-repository/cresci-2017")
        assert "social_spambot_1" in (records[0].notes or "")

    def test_suffixed_classes_are_matched_by_prefix(self, tmp_path: Path):
        path = tmp_path / "cresci.csv"
        path.write_text(CSV_STYLE, encoding="utf-8")
        records = read_label_table(path, dataset="d")
        assert records[0].label is Label.AUTOMATED
        assert records[1].label is Label.ORGANIC

    def test_a_header_is_detected_and_skipped(self, tmp_path: Path):
        path = tmp_path / "labels.csv"
        path.write_text(CSV_STYLE, encoding="utf-8")
        records = read_label_table(path, dataset="d")
        assert [r.account_id for r in records] == ["u1", "u2"]

    def test_explicit_column_names_are_honoured(self, tmp_path: Path):
        path = tmp_path / "labels.csv"
        path.write_text("clase,cuenta\nbot,u9\n", encoding="utf-8")
        records = read_label_table(path, dataset="d", id_column="cuenta", class_column="clase")
        assert records[0].account_id == "u9"
        assert records[0].label is Label.AUTOMATED

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / "labels.dat"
        path.write_text("u1\tbot\n\nu2\thuman\n\n", encoding="utf-8")
        assert len(read_label_table(path, dataset="d")) == 2

    def test_an_empty_file_yields_nothing(self, tmp_path: Path):
        path = tmp_path / "empty.dat"
        path.write_text("", encoding="utf-8")
        assert read_label_table(path, dataset="d") == ()


class TestRefusalToGuess:
    def test_numeric_classes_need_a_declared_convention(self, tmp_path: Path):
        # 1 means bot in one published file and the opposite convention exists
        # elsewhere. Picking one would invert an entire training set silently.
        path = tmp_path / "numeric.dat"
        path.write_text(NUMERIC_STYLE, encoding="utf-8")
        with pytest.raises(UnknownClassError, match="numeric class"):
            read_label_table(path, dataset="d")

    @pytest.mark.parametrize(
        ("convention", "expected"),
        [("1_is_bot", Label.AUTOMATED), ("0_is_bot", Label.ORGANIC)],
    )
    def test_a_declared_convention_is_applied(
        self, tmp_path: Path, convention: str, expected: Label
    ):
        path = tmp_path / "numeric.dat"
        path.write_text(NUMERIC_STYLE, encoding="utf-8")
        records = read_label_table(path, dataset="d", numeric_convention=convention)  # type: ignore[arg-type]
        assert records[0].label is expected

    def test_an_unknown_class_raises_instead_of_becoming_unknown(self, tmp_path: Path):
        path = tmp_path / "labels.dat"
        path.write_text("u1\tcyborg\n", encoding="utf-8")
        with pytest.raises(UnknownClassError, match="unrecognised class"):
            read_label_table(path, dataset="d")

    def test_a_row_with_one_column_is_an_error_not_a_skip(self, tmp_path: Path):
        path = tmp_path / "labels.dat"
        path.write_text("u1\tbot\nu2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="at least two columns"):
            read_label_table(path, dataset="d")


# -------------------------------------------------------------------- archive


class TestIOArchive:
    def test_a_takedown_loads_with_its_profiles(self, io_tweets: Path, io_users: Path):
        result = IOArchiveAdapter(dataset="twitter-io-archive/2019-06-iran").load_takedown(
            io_tweets, io_users
        )
        assert result.report.n_posts == 3
        assert result.report.n_accounts == 2
        assert result.corpus.accounts_by_id["u1"].handle == "cuenta_uno"
        assert result.corpus.accounts_by_id["u1"].followers_count == 1200

    def test_every_account_is_labelled_as_an_operation_not_as_a_bot(
        self, io_tweets: Path, io_users: Path
    ):
        # The distinction is the point: plenty of these accounts were run by
        # people, by hand, full time.
        result = IOArchiveAdapter(dataset="d").load_takedown(io_tweets, io_users)
        labels = result.corpus.labels
        assert {r.label for r in labels} == {Label.INFO_OPERATION}
        assert {r.method for r in labels} == {LabelMethod.PLATFORM_ENFORCEMENT}

    def test_the_single_class_problem_is_warned_about_at_load_time(self, io_tweets: Path):
        result = IOArchiveAdapter(dataset="d").load(io_tweets)
        assert any("one class" in warning for warning in result.report.warnings)

    def test_the_retweet_flag_becomes_a_post_kind(self, io_tweets: Path):
        corpus = IOArchiveAdapter(dataset="d").load(io_tweets).corpus
        kinds = {post.post_id: post.kind for post in corpus.posts}
        assert kinds["t1"] is PostKind.ORIGINAL
        assert kinds["t2"] is PostKind.REPOST
        assert kinds["t3"] is PostKind.REPLY

    def test_bracketed_hashtag_lists_are_read_not_dropped(self, io_tweets: Path):
        # The archive writes [tag1, tag2], which is not valid JSON. Returning
        # nothing there would silently lose every hashtag in the dataset.
        corpus = IOArchiveAdapter(dataset="d").load(io_tweets).corpus
        first = next(p for p in corpus.posts if p.post_id == "t1")
        assert first.hashtags == ("elecciones", "ahora")

    def test_the_archives_offset_free_timestamps_are_declared_not_guessed(self, io_tweets: Path):
        # The archive ships naive timestamps and documents them as UTC. The
        # adapter declares that once, so nothing is dropped and nothing is
        # guessed per row.
        result = IOArchiveAdapter(dataset="d").load(io_tweets)
        assert result.report.n_skipped == 0
        assert all(post.created_at.tzinfo is UTC for post in result.corpus.posts)

    def test_the_declaration_can_be_overridden(self, io_tweets: Path):
        shifted = IOArchiveAdapter(dataset="d", timezone=timezone(timedelta(hours=3)))
        first = shifted.load(io_tweets).corpus.posts[0]
        baseline = IOArchiveAdapter(dataset="d").load(io_tweets).corpus.posts[0]
        assert first.created_at == baseline.created_at - timedelta(hours=3)

    def test_it_recognises_its_own_filenames(self, io_tweets: Path, tmp_path: Path):
        adapter = IOArchiveAdapter(dataset="d")
        assert adapter.sniff(io_tweets)
        assert not adapter.sniff(tmp_path / "something_else.json")

    def test_the_dataset_name_travels_onto_the_labels(self, io_tweets: Path):
        result = IOArchiveAdapter(dataset="twitter-io-archive/2019-06-iran").load(io_tweets)
        assert all(r.dataset == "twitter-io-archive/2019-06-iran" for r in result.corpus.labels)


# ------------------------------------------------------------------- coverage


class TestCoverage:
    def _corpus(self, n_accounts: int = 4):
        accounts = [make_account(f"a{i}") for i in range(n_accounts)]
        posts = [make_post(f"p{i}", f"a{i}") for i in range(n_accounts)]
        return build_corpus(accounts, posts)

    def _labels(self, tmp_path: Path, body: str, **kwargs: object):
        path = tmp_path / "labels.dat"
        path.write_text(body, encoding="utf-8")
        return read_label_table(path, dataset="d", platform=Platform.GENERIC, **kwargs)  # type: ignore[arg-type]

    def test_a_balanced_label_set_raises_no_warnings(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\na1\tbot\na2\thuman\na3\thuman\n")
        _, coverage = attach_labels(self._corpus(), labels)
        assert coverage.n_matched == 4
        assert coverage.match_rate == 1.0
        assert coverage.warnings == ()
        assert coverage.counts[Label.AUTOMATED] == 2

    def test_a_single_class_set_is_flagged(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\na1\tbot\n")
        _, coverage = attach_labels(self._corpus(), labels)
        assert any("single-class" in w for w in coverage.warnings)
        assert coverage.minority_share == 0.0

    def test_severe_imbalance_is_flagged(self, tmp_path: Path):
        accounts = [make_account(f"a{i}") for i in range(40)]
        posts = [make_post(f"p{i}", f"a{i}") for i in range(40)]
        corpus = build_corpus(accounts, posts)
        rows = "".join(f"a{i}\thuman\n" for i in range(39)) + "a39\tbot\n"
        _, coverage = attach_labels(corpus, self._labels(tmp_path, rows))
        assert any("imbalance" in w for w in coverage.warnings)
        assert 0 < coverage.minority_share < 0.05

    def test_labels_for_absent_accounts_are_counted_not_dropped_quietly(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\nghost1\tbot\nghost2\thuman\nghost3\thuman\n")
        labelled, coverage = attach_labels(self._corpus(), labels)
        assert coverage.n_matched == 1
        assert coverage.n_unmatched == 3
        assert len(labelled.labels) == 1
        assert any("matched an account" in w for w in coverage.warnings)

    def test_mixing_datasets_is_flagged(self, tmp_path: Path):
        first = self._labels(tmp_path, "a0\tbot\na1\thuman\n")
        second = tuple(r.model_copy(update={"dataset": "other"}) for r in first)
        _, coverage = attach_labels(self._corpus(), [*first, *second])
        assert any("different annotation procedures" in w for w in coverage.warnings)
        assert len(coverage.datasets) == 2

    def test_unlabelled_accounts_are_counted(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\na1\thuman\n")
        _, coverage = attach_labels(self._corpus(), labels)
        assert coverage.n_unlabelled_accounts == 2

    def test_attaching_twice_does_not_duplicate(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\na1\thuman\n")
        once, _ = attach_labels(self._corpus(), labels)
        twice, _ = attach_labels(once, labels)
        assert len(twice.labels) == 2

    def test_replace_discards_what_was_there(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\na1\thuman\n")
        once, _ = attach_labels(self._corpus(), labels)
        replacement = tuple(r.model_copy(update={"dataset": "new"}) for r in labels)
        twice, _ = attach_labels(once, replacement, replace=True)
        assert {r.dataset for r in twice.labels} == {"new"}

    def test_coverage_reads_the_corpus_labels_when_none_are_passed(self, io_tweets: Path):
        corpus = IOArchiveAdapter(dataset="d").load(io_tweets).corpus
        coverage = label_coverage(corpus)
        assert coverage.n_matched == len(corpus.labels)
        assert any("single-class" in w for w in coverage.warnings)

    def test_coverage_is_serialisable(self, tmp_path: Path):
        labels = self._labels(tmp_path, "a0\tbot\na1\thuman\n")
        _, coverage = attach_labels(self._corpus(), labels)
        exported = coverage.as_dict()
        assert exported["n_matched"] == 2
        assert exported["counts"] == {"automated": 1, "organic": 1}
        assert exported["match_rate"] == 1.0

    def test_an_empty_label_set_is_not_a_division_by_zero(self):
        coverage = label_coverage(build_corpus([], []), [])
        assert isinstance(coverage, LabelCoverage)
        assert coverage.match_rate == 0.0
        assert coverage.minority_share == 0.0


def test_io_archive_dates_are_normalised_to_utc(io_tweets: Path, io_users: Path):
    corpus = IOArchiveAdapter(dataset="d").load_takedown(io_tweets, io_users).corpus
    created = corpus.accounts_by_id["u1"].created_at
    assert created is not None
    assert created.tzinfo is UTC
    assert created == datetime(2016, 4, 1, tzinfo=UTC)


class TestLabelsCommand:
    def test_it_reports_coverage_and_writes_the_labelled_corpus(self, tmp_path: Path):
        accounts = [make_account(f"a{i}") for i in range(4)]
        posts = [make_post(f"p{i}", f"a{i}") for i in range(4)]
        corpus_file = write_corpus(build_corpus(accounts, posts), tmp_path / "corpus.json")
        annotations = tmp_path / "labels.dat"
        annotations.write_text("a0\tbot\na1\tbot\na2\thuman\na3\thuman\n", encoding="utf-8")
        out = tmp_path / "labelled.json"

        code = main(
            [
                "labels",
                str(corpus_file),
                str(annotations),
                "--dataset",
                "indiana-bot-repository/varol-2017",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        assert out.exists()
        assert '"automated"' in out.read_text(encoding="utf-8")

    def test_an_ambiguous_numeric_file_fails_loudly(self, tmp_path: Path):
        corpus_file = write_corpus(
            build_corpus([make_account("a0")], [make_post("p0", "a0")]), tmp_path / "c.json"
        )
        annotations = tmp_path / "labels.dat"
        annotations.write_text("a0\t1\n", encoding="utf-8")
        with pytest.raises(UnknownClassError, match="numeric class"):
            main(["labels", str(corpus_file), str(annotations), "--dataset", "d"])


# The exact header of ioa_tweets.csv in the Internet Archive mirror of X's
# information operations disclosures, with rows shaped like the real ones.
REAL_TWEETS = (
    "tweetid,userid,user_display_name,user_screen_name,user_reported_location,"
    "user_profile_description,user_profile_url,follower_count,following_count,"
    "account_creation_date,account_language,tweet_language,tweet_text,tweet_time,"
    "tweet_client_name,in_reply_to_userid,in_reply_to_tweetid,quoted_tweet_tweetid,"
    "is_retweet,retweet_userid,retweet_tweetid,latitude,longitude,quote_count,"
    "reply_count,like_count,retweet_count,hashtags,urls,user_mentions,poll_choices\n"
    "898925911294132224,ygQRwhQRrh1+6N7J6IMFnzWgqUGimWqg0KZptLpxDY=,"
    "ygQRwhQRrh1+6N7J6IMFnzWgqUGimWqg0KZptLpxDY=,"
    "ygQRwhQRrh1+6N7J6IMFnzWgqUGimWqg0KZptLpxDY=,"
    '"Dhaka, Bangladesh",Only news portal,,17,23,2016-12-04,en,bn,'
    "Un mensaje del archivo real,2017-08-19 15:13,dlvr.it,,,,False,,,absent,absent,"
    "0.0,0.0,0.0,0.0,[],['http://dlvr.it/PgB3PP'],[],\n"
    "861871665780846592,0OvJRirnL32w7blCoMRthk30yPlXapfIPFac4Eyje0k=,"
    "0OvJRirnL32w7blCoMRthk30yPlXapfIPFac4Eyje0k=,"
    "0OvJRirnL32w7blCoMRthk30yPlXapfIPFac4Eyje0k=,"
    "Bangladesh,FB link,,758,11,2017-05-08,en,bn,"
    "Otro mensaje del archivo real,2017-05-09 09:41,Twitter Web Client,,,,True,,,"
    "absent,absent,0.0,0.0,0.0,0.0,\"['uno', 'dos']\",[],[],\n"
)


@pytest.fixture
def real_tweets(tmp_path: Path) -> Path:
    path = tmp_path / "ioa_tweets.csv"
    path.write_text(REAL_TWEETS, encoding="utf-8")
    return path


class TestRealArchiveShape:
    """Against the actual column layout of the published archive."""

    def test_profiles_are_derived_when_no_users_file_is_given(self, real_tweets: Path):
        # The consolidated file repeats every profile column on each tweet row.
        result = IOArchiveAdapter(dataset="x-io/consolidated").load(real_tweets)
        assert result.report.n_posts == 2
        assert result.report.n_accounts == 2
        account = result.corpus.accounts_by_id["ygQRwhQRrh1+6N7J6IMFnzWgqUGimWqg0KZptLpxDY="]
        assert account.followers_count == 17
        assert account.location == "Dhaka, Bangladesh"
        assert account.created_at is not None

    def test_python_repr_lists_are_read(self, real_tweets: Path):
        # The archive writes ['a', 'b'], which is not JSON.
        corpus = IOArchiveAdapter(dataset="d").load(real_tweets).corpus
        first = next(p for p in corpus.posts if p.post_id == "898925911294132224")
        second = next(p for p in corpus.posts if p.post_id == "861871665780846592")
        assert first.urls == ("http://dlvr.it/PgB3PP",)
        assert second.hashtags == ("uno", "dos")

    def test_the_placeholder_geo_values_land_in_extra(self, real_tweets: Path):
        corpus = IOArchiveAdapter(dataset="d").load(real_tweets).corpus
        assert corpus.posts[0].extra["latitude"] == "absent"

    def test_float_counts_are_read_as_integers(self, real_tweets: Path):
        corpus = IOArchiveAdapter(dataset="d").load(real_tweets).corpus
        assert corpus.posts[0].like_count == 0

    def test_everything_is_labelled_once(self, real_tweets: Path):
        corpus = IOArchiveAdapter(dataset="d").load(real_tweets).corpus
        assert len(corpus.labels) == 2
        assert {r.label for r in corpus.labels} == {Label.INFO_OPERATION}


class TestAnalyseWithTheArchiveAdapter:
    """Without it, an archive export loads as posts with no accounts at all."""

    def test_the_default_adapter_builds_no_accounts(self, real_tweets: Path, tmp_path: Path):
        out = tmp_path / "native.json"
        main(["analyse", str(real_tweets), "--assume-timezone", "0", "--json", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["corpus"]["n_accounts"] == 0
        assert payload["corpus"]["orphan_posts"] == 2

    def test_the_archive_adapter_derives_them_from_the_tweet_rows(
        self, real_tweets: Path, tmp_path: Path
    ):
        out = tmp_path / "archive.json"
        main(["analyse", str(real_tweets), "--adapter", "io-archive", "--json", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["corpus"]["n_accounts"] == 2
        assert payload["corpus"]["orphan_posts"] == 0

    def test_the_profile_features_then_have_something_to_measure(
        self, real_tweets: Path, tmp_path: Path
    ):
        out = tmp_path / "features.csv"
        main(
            [
                "analyse",
                str(real_tweets),
                "--adapter",
                "io-archive",
                "--features",
                str(out),
            ]
        )
        body = out.read_text(encoding="utf-8")
        assert "acct_followback_ratio" in body
        # Two real rows, not an empty frame.
        assert len(body.strip().splitlines()) == 3
