"""Tests for the file profiler.

Its job is to answer, before anyone commits to a load, what a real dataset will
do: which columns are understood, which are not, and what a trial run skips.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any, cast

import pytest

from synthwatch.cli import main
from synthwatch.ingest.inspect import inspect_file, render_profile
from synthwatch.ingest.native import NativeAdapter

# The header of the published archive of X's information operations, with rows
# in the shape the real file uses -- ambiguous US dates and all.
ARCHIVE_USERS = (
    "userid,user_display_name,user_screen_name,user_reported_location,"
    "user_profile_description,user_profile_url,follower_count,following_count,"
    "account_creation_date,account_language\n"
    "1.01421E+18,Khela,Khela12667528,,,,,22,7/3/2018,en\n"
    "MoOvnxuLWFdv+xY3Q8354PM4Zv2KhI2Y5V+jfwCnU0=,Juger,juger_ch,"
    '"Narayanganj, Bangladesh",Noticias,https://t.co/x,4,7,8/11/2018,en\n'
    "SejILTNvsopgs7LAojRXwXG8gzlQZGaRxTefFneouI=,News,news_bd,"
    '"Dhaka, Bangladesh",Servicio de noticias,,17,25,8/28/2016,en\n'
)

PLAIN_POSTS = (
    "post_id,account_id,created_at,text,subreddit\n"
    "p1,a1,2024-03-01T12:00:00Z,Un mensaje,politica\n"
    "p2,a2,2024-03-01T12:05:00Z,Otro mensaje,politica\n"
)


@pytest.fixture
def archive_users(tmp_path: Path) -> Path:
    path = tmp_path / "ioa_users.csv"
    path.write_text(ARCHIVE_USERS, encoding="utf-8")
    return path


@pytest.fixture
def plain_posts(tmp_path: Path) -> Path:
    path = tmp_path / "posts.csv"
    path.write_text(PLAIN_POSTS, encoding="utf-8")
    return path


class TestProfile:
    def test_it_reports_every_column_in_file_order(self, plain_posts: Path):
        profile = inspect_file(plain_posts)
        assert [c.name for c in profile.columns] == [
            "post_id",
            "account_id",
            "created_at",
            "text",
            "subreddit",
        ]

    def test_it_separates_understood_columns_from_the_rest(self, plain_posts: Path):
        profile = inspect_file(plain_posts)
        assert "post_id" in profile.recognised
        assert profile.unrecognised == ("subreddit",)

    def test_it_samples_example_values(self, plain_posts: Path):
        profile = inspect_file(plain_posts)
        text = next(c for c in profile.columns if c.name == "text")
        assert text.examples[0] == "Un mensaje"

    def test_it_measures_how_often_a_column_is_filled(self, archive_users: Path):
        profile = inspect_file(archive_users, kind="accounts")
        url = next(c for c in profile.columns if c.name == "user_profile_url")
        assert url.fill_rate == pytest.approx(1 / 3)

    def test_a_clean_file_is_usable(self, plain_posts: Path):
        profile = inspect_file(plain_posts)
        assert profile.usable
        assert profile.n_trial_loaded == 2
        assert profile.trial_skips == {}

    def test_a_file_without_the_required_fields_is_not_usable(self, tmp_path: Path):
        path = tmp_path / "mystery.csv"
        path.write_text("colA,colB\n1,2\n", encoding="utf-8")
        profile = inspect_file(path)
        assert not profile.usable
        assert "post_id" in profile.missing_required

    def test_it_is_serialisable(self, plain_posts: Path):
        exported = inspect_file(plain_posts).as_dict()
        assert exported["usable"] is True
        columns = cast("list[dict[str, Any]]", exported["columns"])
        assert columns[0]["name"] == "post_id"

    def test_the_sample_size_is_respected(self, tmp_path: Path):
        path = tmp_path / "many.csv"
        rows = "".join(f"p{n},a1,2024-03-01T12:00:00Z,hola\n" for n in range(500))
        path.write_text(f"post_id,account_id,created_at,text\n{rows}", encoding="utf-8")
        assert inspect_file(path, sample_rows=50).n_sampled == 50


class TestAgainstTheRealArchiveShape:
    """The profiler is the tool for meeting a dataset for the first time."""

    def test_it_finds_the_spreadsheet_corrupted_identifiers(self, archive_users: Path):
        # One row in three here has had its 18-digit id rounded to 1.01421E+18.
        profile = inspect_file(archive_users, kind="accounts")
        assert profile.trial_skips["account:corrupted_identifier"] == 1

    def test_it_reports_the_columns_the_defaults_do_not_know(self, archive_users: Path):
        profile = inspect_file(archive_users, kind="accounts")
        assert "account_creation_date" in profile.unrecognised

    def test_an_alias_brings_a_column_in(self, archive_users: Path):
        adapter = NativeAdapter(account_aliases={"account_creation_date": "created_at"})
        profile = inspect_file(archive_users, kind="accounts", adapter=adapter)
        assert "created_at" in profile.recognised
        # ...and immediately surfaces that those dates are ambiguous.
        assert profile.trial_skips["account:ambiguous_date"] >= 1

    def test_declaring_both_the_order_and_the_timezone_recovers_the_rows(self, archive_users: Path):
        adapter = NativeAdapter(
            account_aliases={"account_creation_date": "created_at"},
            date_order="mdy",
            assume_timezone=UTC,
        )
        profile = inspect_file(archive_users, kind="accounts", adapter=adapter)
        assert profile.n_trial_loaded == 2
        assert set(profile.trial_skips) == {"account:corrupted_identifier"}


class TestRendering:
    def test_the_text_view_names_what_was_skipped(self, archive_users: Path):
        rendered = render_profile(inspect_file(archive_users, kind="accounts"))
        assert "corrupted_identifier" in rendered
        assert "-- extra --" in rendered

    def test_it_tells_you_what_is_missing(self, tmp_path: Path):
        path = tmp_path / "mystery.csv"
        path.write_text("colA,colB\n1,2\n", encoding="utf-8")
        rendered = render_profile(inspect_file(path))
        assert "MISSING required field(s)" in rendered
        assert "--alias" in rendered

    def test_a_clean_file_says_so(self, plain_posts: Path):
        assert "nothing skipped" in render_profile(inspect_file(plain_posts))


class TestInspectCommand:
    def test_it_exits_zero_on_a_usable_file(self, plain_posts: Path, capsys):
        assert main(["inspect", str(plain_posts)]) == 0
        assert "maps to" in capsys.readouterr().out

    def test_it_exits_nonzero_when_the_file_cannot_be_loaded(self, tmp_path: Path):
        path = tmp_path / "mystery.csv"
        path.write_text("colA,colB\n1,2\n", encoding="utf-8")
        assert main(["inspect", str(path)]) == 1

    def test_aliases_are_applied_from_the_command_line(self, archive_users: Path, tmp_path: Path):
        out = tmp_path / "profile.json"
        main(
            [
                "inspect",
                str(archive_users),
                "--kind",
                "accounts",
                "--alias",
                "account_creation_date=created_at",
                "--date-order",
                "mdy",
                "--assume-timezone",
                "0",
                "--json",
                str(out),
            ]
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["n_trial_loaded"] == 2
        assert "created_at" in payload["recognised"]

    def test_a_malformed_alias_is_refused(self, plain_posts: Path):
        with pytest.raises(ValueError, match="COLUMN=FIELD"):
            main(["inspect", str(plain_posts), "--alias", "nonsense"])
