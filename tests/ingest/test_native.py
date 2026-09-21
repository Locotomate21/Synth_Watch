"""Tests for the native CSV/JSON/JSONL adapter.

Two things carry most of the weight here: a bad row must be *counted*, never
lost, and a timestamp without an offset must never be quietly assumed to be
UTC.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from synthwatch.detect.account import AccountConfig
from synthwatch.detect.coordination import detect_coordination
from synthwatch.ingest import REGISTRY
from synthwatch.ingest.native import (
    AmbiguousDateError,
    CorruptedIdentifierError,
    NaiveTimestampError,
    NativeAdapter,
    check_identifier,
    parse_timestamp,
    timezone_of,
    write_corpus,
)
from synthwatch.models import Post, build_corpus
from synthwatch.types import Platform, PostKind
from tests.conftest import EPOCH, make_account, make_post
from tests.detect.fixtures_coordination import coordinated_corpus

POSTS_CSV = """post_id,account_id,created_at,text,kind,hashtags
p1,a1,2024-03-01T12:00:00+00:00,Primer mensaje de prueba,original,uno|dos
p2,a2,2024-03-01T12:05:00+00:00,Segundo mensaje de prueba,reply,
p3,a1,2024-03-01T13:00:00Z,Tercer mensaje de prueba,original,tres
"""

ACCOUNTS_CSV = """account_id,handle,created_at,followers,following,verified
a1,@primera,2020-01-01T00:00:00+00:00,1200,340,true
a2,segunda,2021-06-15T00:00:00+00:00,15,900,false
"""


@pytest.fixture
def posts_csv(tmp_path: Path) -> Path:
    path = tmp_path / "posts.csv"
    path.write_text(POSTS_CSV, encoding="utf-8")
    return path


@pytest.fixture
def accounts_csv(tmp_path: Path) -> Path:
    path = tmp_path / "accounts.csv"
    path.write_text(ACCOUNTS_CSV, encoding="utf-8")
    return path


# ------------------------------------------------------------------ timestamps


class TestTimestamps:
    def test_iso_with_offset(self):
        parsed = parse_timestamp("2024-03-01T14:00:00+02:00")
        assert parsed == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    def test_trailing_z_is_accepted(self):
        assert parse_timestamp("2024-03-01T12:00:00Z").tzinfo is not None

    def test_epoch_seconds(self):
        assert parse_timestamp(1709294400) == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    def test_epoch_milliseconds(self):
        assert parse_timestamp(1709294400000) == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    def test_legacy_twitter_format(self):
        parsed = parse_timestamp("Fri Mar 01 12:00:00 +0000 2024")
        assert parsed == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    def test_naive_timestamp_is_refused_by_default(self):
        # The whole point: a guessed offset is invisible downstream.
        with pytest.raises(NaiveTimestampError, match="no UTC offset"):
            parse_timestamp("2024-03-01 12:00:00")

    def test_naive_timestamp_is_accepted_when_declared(self):
        parsed = parse_timestamp("2024-03-01 12:00:00", assume_timezone=timezone_of(-5))
        assert parsed == datetime(2024, 3, 1, 17, 0, tzinfo=UTC)

    def test_unparsable_timestamp_raises(self):
        with pytest.raises(ValueError, match="unrecognised timestamp"):
            parse_timestamp("el martes pasado")

    def test_empty_timestamp_raises(self):
        with pytest.raises(ValueError, match="empty timestamp"):
            parse_timestamp("   ")


# ----------------------------------------------------------------------- csv


class TestCsv:
    def test_posts_and_accounts_load_together(self, posts_csv: Path, accounts_csv: Path):
        result = NativeAdapter().load_tables(posts_csv, accounts_csv)
        assert result.report.n_posts == 3
        assert result.report.n_accounts == 2
        assert result.report.n_skipped == 0
        assert result.corpus.orphan_post_ids == ()

    def test_values_are_typed_not_left_as_strings(self, posts_csv: Path, accounts_csv: Path):
        corpus = NativeAdapter().load_tables(posts_csv, accounts_csv).corpus
        account = corpus.accounts_by_id["a1"]
        assert account.followers_count == 1200
        assert account.verified is True
        assert account.handle == "primera"  # the @ is stripped by the schema
        post = corpus.posts_by_account["a2"][0]
        assert post.kind is PostKind.REPLY

    def test_list_columns_are_split(self, posts_csv: Path):
        corpus = NativeAdapter().load(posts_csv).corpus
        assert corpus.posts_by_account["a1"][0].hashtags == ("uno", "dos")
        assert corpus.posts_by_account["a2"][0].hashtags == ()

    def test_timestamps_are_normalised_to_utc(self, posts_csv: Path):
        corpus = NativeAdapter().load(posts_csv).corpus
        assert all(post.created_at.tzinfo is UTC for post in corpus.posts)

    def test_platform_is_stamped_from_the_adapter(self, posts_csv: Path):
        corpus = NativeAdapter(platform=Platform.MASTODON).load(posts_csv).corpus
        assert {post.platform for post in corpus.posts} == {Platform.MASTODON}

    def test_a_platform_column_wins_over_the_default(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text,platform\n"
            "p1,a1,2024-03-01T12:00:00Z,hola que tal,reddit\n",
            encoding="utf-8",
        )
        corpus = NativeAdapter(platform=Platform.GENERIC).load(path).corpus
        assert corpus.posts[0].platform is Platform.REDDIT

    def test_unknown_columns_survive_in_extra(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text,subreddit,score\n"
            "p1,a1,2024-03-01T12:00:00Z,hola,politica,42\n",
            encoding="utf-8",
        )
        post = NativeAdapter().load(path).corpus.posts[0]
        assert post.extra == {"subreddit": "politica", "score": "42"}

    def test_tsv_is_read_with_tabs(self, tmp_path: Path):
        path = tmp_path / "posts.tsv"
        path.write_text(
            "post_id\taccount_id\tcreated_at\ttext\np1\ta1\t2024-03-01T12:00:00Z\tcon tabs\n",
            encoding="utf-8",
        )
        assert NativeAdapter().load(path).corpus.posts[0].text == "con tabs"

    def test_bom_does_not_break_the_header(self, tmp_path: Path):
        # Excel writes UTF-8 with a BOM; without utf-8-sig the first column name
        # becomes "﻿post_id" and every row silently loses its id.
        path = tmp_path / "posts.csv"
        path.write_text(
            "﻿post_id,account_id,created_at,text\np1,a1,2024-03-01T12:00:00Z,hola\n",
            encoding="utf-8",
        )
        assert NativeAdapter().load(path).report.n_posts == 1


# -------------------------------------------------------------------- aliases


class TestAliases:
    def test_common_export_columns_are_recognised(self, tmp_path: Path):
        path = tmp_path / "archive.csv"
        path.write_text(
            "tweetid,userid,tweet_time,tweet_text,retweet_count\n"
            "9001,u7,2024-03-01T12:00:00Z,Mensaje del archivo,12\n",
            encoding="utf-8",
        )
        post = NativeAdapter().load(path).corpus.posts[0]
        assert post.post_id == "9001"
        assert post.account_id == "u7"
        assert post.repost_count == 12

    def test_a_canonical_column_beats_an_alias(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "id,post_id,account_id,created_at,text\nwrong,right,a1,2024-03-01T12:00:00Z,hola\n",
            encoding="utf-8",
        )
        assert NativeAdapter().load(path).corpus.posts[0].post_id == "right"

    def test_custom_aliases_are_merged_in(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "identificador,autor,fecha,contenido\n"
            "p1,a1,2024-03-01T12:00:00Z,Contenido en castellano\n",
            encoding="utf-8",
        )
        adapter = NativeAdapter(
            post_aliases={
                "identificador": "post_id",
                "autor": "account_id",
                "fecha": "created_at",
                "contenido": "text",
            }
        )
        post = adapter.load(path).corpus.posts[0]
        assert (post.post_id, post.account_id) == ("p1", "a1")


# ---------------------------------------------------------------- resilience


class TestBadRows:
    def test_bad_rows_are_counted_not_lost(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\n"
            "p1,a1,2024-03-01T12:00:00Z,valida\n"
            "p2,a1,el martes pasado,fecha ilegible\n"
            "p3,a1,2024-03-01 12:00:00,sin offset\n"
            ",a1,2024-03-01T12:00:00Z,sin id\n",
            encoding="utf-8",
        )
        report = NativeAdapter().load(path).report
        assert report.n_posts == 1
        assert report.n_skipped == 3
        assert report.skip_reasons["post:naive_timestamp"] == 1
        assert sum(report.skip_reasons.values()) == 3

    def test_a_high_drop_rate_is_reported_as_a_warning(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        rows = "".join(f"p{n},a1,no es una fecha,texto\n" for n in range(5))
        path.write_text(f"post_id,account_id,created_at,text\n{rows}", encoding="utf-8")
        report = NativeAdapter().load(path).report
        assert any("drop rate" in warning for warning in report.warnings)

    def test_strict_mode_raises_on_the_first_bad_row(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\np1,a1,2024-03-01 12:00:00,sin offset\n",
            encoding="utf-8",
        )
        with pytest.raises(NaiveTimestampError):
            NativeAdapter(strict=True).load(path)

    def test_orphan_posts_are_warned_about_not_dropped(self, posts_csv: Path, tmp_path: Path):
        accounts = tmp_path / "accounts.csv"
        accounts.write_text("account_id,handle\na1,primera\n", encoding="utf-8")
        result = NativeAdapter().load_tables(posts_csv, accounts)
        assert result.corpus.orphan_post_ids == ("p2",)
        assert any("orphan" in warning for warning in result.report.warnings)

    def test_malformed_jsonl_lines_are_counted(self, tmp_path: Path):
        path = tmp_path / "posts.jsonl"
        path.write_text(
            json.dumps({"post_id": "p1", "account_id": "a1", "created_at": "2024-03-01T12:00:00Z"})
            + "\n{ esto no es json }\n\n",
            encoding="utf-8",
        )
        report = NativeAdapter().load(path).report
        assert report.n_posts == 1
        assert report.skip_reasons["post:malformed_json"] == 1


# ------------------------------------------------------------------ formats


class TestFormats:
    def test_jsonl_posts(self, tmp_path: Path):
        path = tmp_path / "posts.jsonl"
        lines = [
            {
                "post_id": f"p{n}",
                "account_id": "a1",
                "created_at": "2024-03-01T12:00:00Z",
                "text": "hola",
                "hashtags": ["uno", "dos"],
            }
            for n in range(3)
        ]
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
        corpus = NativeAdapter().load(path).corpus
        assert len(corpus) == 3
        assert corpus.posts[0].hashtags == ("uno", "dos")

    def test_json_object_with_all_three_tables(self, tmp_path: Path):
        path = tmp_path / "corpus.json"
        path.write_text(
            json.dumps(
                {
                    "accounts": [{"account_id": "a1", "handle": "primera"}],
                    "posts": [
                        {
                            "post_id": "p1",
                            "account_id": "a1",
                            "created_at": "2024-03-01T12:00:00Z",
                            "text": "hola",
                        }
                    ],
                    "labels": [
                        {
                            "account_id": "a1",
                            "label": "automated",
                            "dataset": "indiana-bot-repository/cresci-2017",
                            "method": "human_annotation",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        result = NativeAdapter().load(path)
        assert (result.report.n_accounts, result.report.n_posts) == (1, 1)
        assert result.corpus.labels_by_account["a1"].dataset.startswith("indiana")

    def test_bare_json_array_is_read_as_posts(self, tmp_path: Path):
        path = tmp_path / "posts.json"
        path.write_text(
            json.dumps(
                [{"post_id": "p1", "account_id": "a1", "created_at": "2024-03-01T12:00:00Z"}]
            ),
            encoding="utf-8",
        )
        assert NativeAdapter().load(path).report.n_posts == 1

    def test_unsupported_suffix_is_refused(self, tmp_path: Path):
        path = tmp_path / "posts.xlsx"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported file type"):
            NativeAdapter().load(path)


# ------------------------------------------------------------------ round trip


class TestRoundTrip:
    def test_a_corpus_survives_write_and_read(self, tmp_path: Path):
        original = coordinated_corpus()
        path = write_corpus(original, tmp_path / "corpus.json")
        restored = NativeAdapter().load(path).corpus

        assert len(restored) == len(original)
        assert len(restored.accounts) == len(original.accounts)
        assert [p.post_id for p in restored.posts] == [p.post_id for p in original.posts]
        assert [p.text for p in restored.posts] == [p.text for p in original.posts]
        assert [p.created_at for p in restored.posts] == [p.created_at for p in original.posts]

    def test_round_trip_preserves_extra_and_counts(self, tmp_path: Path):
        original = make_post("p1", "a1", text="hola que tal", extra={"subreddit": "politica"})
        account = make_account("a1", followers=1234)
        path = write_corpus(build_corpus([account], [original]), tmp_path / "corpus.json")
        restored = NativeAdapter().load(path).corpus
        assert restored.posts[0].extra == {"subreddit": "politica"}
        assert restored.accounts_by_id["a1"].followers_count == 1234

    def test_the_analysis_survives_a_round_trip(self, tmp_path: Path):
        # End to end: the planted cluster must still be found after the corpus
        # has been through a file.
        path = write_corpus(coordinated_corpus(), tmp_path / "corpus.json")
        restored = NativeAdapter().load(path).corpus
        clusters = detect_coordination(restored).clusters
        assert len(clusters) == 1
        assert set(clusters[0].account_ids) == {"sync0", "sync1", "sync2", "sync3"}


# -------------------------------------------------------------------- registry


class TestRegistry:
    def test_the_native_adapter_is_registered_on_import(self):
        assert "native" in REGISTRY.names
        assert REGISTRY.get("native").platform is Platform.GENERIC

    def test_detection_by_suffix(self, posts_csv: Path):
        detected = REGISTRY.detect(posts_csv)
        assert detected is not None
        assert detected.name == "native"

    def test_unknown_suffix_is_not_detected(self, tmp_path: Path):
        assert REGISTRY.detect(tmp_path / "data.parquet") is None

    def test_unknown_adapter_name_lists_what_exists(self):
        with pytest.raises(KeyError, match="registered: native"):
            REGISTRY.get("facebook")


def test_report_is_serialisable(posts_csv: Path, accounts_csv: Path):
    exported = NativeAdapter().load_tables(posts_csv, accounts_csv).report.as_dict()
    assert exported["n_posts"] == 3
    assert exported["skip_reasons"] == {}


def test_assume_timezone_shifts_the_whole_file(tmp_path: Path):
    path = tmp_path / "posts.csv"
    path.write_text(
        "post_id,account_id,created_at,text\np1,a1,2024-03-01 12:00:00,hola\n",
        encoding="utf-8",
    )
    adapter = NativeAdapter(assume_timezone=timezone(timedelta(hours=2)))
    post: Post = adapter.load(path).corpus.posts[0]
    assert post.created_at == datetime(2024, 3, 1, 10, 0, tzinfo=UTC)


class TestAmbiguousDates:
    def test_a_slash_date_that_could_go_either_way_is_refused(self):
        # 7/3/2018 is the 7th of March or the 3rd of July. Four months apart,
        # and nothing downstream would look wrong.
        with pytest.raises(AmbiguousDateError, match="ambiguous date"):
            parse_timestamp("7/3/2018", assume_timezone=UTC)

    @pytest.mark.parametrize(
        ("order", "month"),
        [("dmy", 3), ("mdy", 7)],
    )
    def test_a_declared_order_is_applied(self, order: str, month: int):
        parsed = parse_timestamp("7/3/2018", assume_timezone=UTC, date_order=order)  # type: ignore[arg-type]
        assert parsed.month == month

    def test_an_unambiguous_day_first_date_needs_no_declaration(self):
        # 13 cannot be a month.
        assert parse_timestamp("13/7/2018", assume_timezone=UTC).day == 13

    def test_an_unambiguous_month_first_date_needs_no_declaration(self):
        assert parse_timestamp("7/13/2018", assume_timezone=UTC).day == 13

    def test_a_time_after_the_date_is_kept(self):
        parsed = parse_timestamp("13/7/2018 21:45", assume_timezone=UTC)
        assert (parsed.hour, parsed.minute) == (21, 45)

    def test_ambiguous_rows_are_counted_with_their_own_reason(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\np1,a1,7/3/2018 10:00,hola\n",
            encoding="utf-8",
        )
        report = NativeAdapter().load(path).report
        assert report.skip_reasons["post:ambiguous_date"] == 1

    def test_declaring_the_order_on_the_adapter_recovers_them(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\np1,a1,7/3/2018 10:00,hola\n",
            encoding="utf-8",
        )
        # Two separate declarations: the date order says how to read 7/3, the
        # timezone says what the clock means. Neither implies the other.
        adapter = NativeAdapter(date_order="mdy", assume_timezone=UTC)
        assert adapter.load(path).corpus.posts[0].created_at.month == 7

    def test_the_date_order_alone_does_not_declare_the_timezone(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\np1,a1,7/3/2018 10:00,hola\n",
            encoding="utf-8",
        )
        report = NativeAdapter(date_order="mdy").load(path).report
        assert report.skip_reasons["post:naive_timestamp"] == 1


class TestCorruptedIdentifiers:
    @pytest.mark.parametrize("value", ["1.01421E+18", "9.8765e+17", "1E+18"])
    def test_scientific_notation_ids_are_refused(self, value: str):
        # A spreadsheet round-trip turns an 18-digit id into this. It looks like
        # data and joins to nothing.
        with pytest.raises(CorruptedIdentifierError, match="scientific notation"):
            check_identifier(value, field="account_id")

    @pytest.mark.parametrize(
        "value",
        ["898925911294132224", "ygQRwhQRrh1+6N7J6IMFnzWgqUGimWqg0KZptLpxDY=", "user_42"],
    )
    def test_real_identifiers_pass(self, value: str):
        assert check_identifier(value, field="account_id") == value

    def test_corrupted_rows_are_counted_with_their_own_reason(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\n"
            "p1,1.01421E+18,2024-03-01T12:00:00Z,hola\n"
            "p2,a2,2024-03-01T12:00:00Z,adios\n",
            encoding="utf-8",
        )
        report = NativeAdapter().load(path).report
        assert report.n_posts == 1
        assert report.skip_reasons["post:corrupted_identifier"] == 1


class TestRecordSeams:
    def test_rows_can_be_read_and_mapped_separately(self, posts_csv: Path):
        # The seam every platform adapter builds on.
        adapter = NativeAdapter()
        rows = adapter.read_records(posts_csv)
        assert len(rows) == 3
        result = adapter.from_records("in-memory", rows)
        assert result.report.n_posts == 3
        assert result.report.source == "in-memory"

    def test_rearranged_rows_still_go_through_the_same_accounting(self):
        adapter = NativeAdapter()
        rows = [{"post_id": "p1", "account_id": "a1", "created_at": "nope", "text": "x"}]
        report = adapter.from_records("in-memory", rows).report
        assert report.n_skipped == 1


class TestStructuralJunk:
    def test_a_blank_row_gets_its_own_reason(self, tmp_path: Path):
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\np1,a1,2024-03-01T12:00:00Z,hola\n,,,\n",
            encoding="utf-8",
        )
        report = NativeAdapter().load(path).report
        assert report.n_posts == 1
        assert report.skip_reasons["post:empty_row"] == 1

    def test_a_header_repeated_inside_the_file_is_recognised(self, tmp_path: Path):
        # Concatenating per-takedown exports without stripping their headers
        # leaves one in the body. Real consolidated archives contain these.
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\n"
            "p1,a1,2024-03-01T12:00:00Z,hola\n"
            "post_id,account_id,created_at,text\n"
            "p2,a2,2024-03-01T12:05:00Z,adios\n",
            encoding="utf-8",
        )
        report = NativeAdapter().load(path).report
        assert report.n_posts == 2
        assert report.skip_reasons["post:repeated_header"] == 1

    def test_a_row_that_merely_resembles_its_header_is_not_discarded(self, tmp_path: Path):
        # Only a row where *every* populated cell equals its column name counts.
        path = tmp_path / "posts.csv"
        path.write_text(
            "post_id,account_id,created_at,text\npost_id,a1,2024-03-01T12:00:00Z,hola\n",
            encoding="utf-8",
        )
        assert NativeAdapter().load(path).report.n_posts == 1


class TestProvenanceSurvivesTheRoundTrip:
    def test_the_collection_time_comes_back(self, tmp_path: Path):
        # Without it there is no reference instant, and every feature measured
        # against one -- account age, dormancy, lifetime posting rate -- goes
        # quietly unmeasurable.
        original = build_corpus(
            [make_account("a1")],
            [make_post("p1", "a1")],
            source="somewhere/specific",
            collected_at=datetime(2024, 3, 1, 12, 0, tzinfo=UTC),
        )
        path = write_corpus(original, tmp_path / "corpus.json")
        restored = NativeAdapter().load(path).corpus
        assert restored.collected_at == original.collected_at
        assert restored.source == "somewhere/specific"

    def test_a_corpus_without_provenance_still_loads(self, tmp_path: Path):
        path = write_corpus(build_corpus([make_account("a1")], []), tmp_path / "corpus.json")
        restored = NativeAdapter().load(path).corpus
        assert restored.collected_at is None
        assert restored.source == path.name

    def test_the_reference_instant_survives_too(self, tmp_path: Path):
        original = build_corpus(
            [make_account("a1", created_days_ago=100)],
            [],
            collected_at=EPOCH,
        )
        path = write_corpus(original, tmp_path / "corpus.json")
        restored = NativeAdapter().load(path).corpus
        assert AccountConfig().reference_for(restored) == EPOCH
