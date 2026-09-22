"""Tests for the report layer and the CLI.

Most of these assert absences: no account-level score in an export, no real
handle in a pseudonymised report, no figure without the parameters that
produced it. The report is the artefact that leaves the machine, so its
restraint is the part worth testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthwatch.cli import main
from synthwatch.detect.account import AccountExtractor
from synthwatch.detect.coordination import detect_coordination
from synthwatch.ingest.native import NativeAdapter, write_corpus
from synthwatch.report import Pseudonymiser, build_cards, build_report, to_html, to_json
from synthwatch.report.report import CORPUS_CAVEATS
from synthwatch.report.summary import summarise_features
from tests.detect.fixtures_coordination import coordinated_corpus, organic_corpus

PLANTED = {"sync0", "sync1", "sync2", "sync3"}


@pytest.fixture(scope="module")
def report_and_features():
    return build_report(coordinated_corpus(), title="Test run")


# ------------------------------------------------------------------- pseudonyms


class TestPseudonymiser:
    def test_the_same_account_always_gets_the_same_pseudonym(self):
        mask = Pseudonymiser(salt="fixed")
        assert mask("sync0") == mask("sync0")

    def test_different_accounts_get_different_pseudonyms(self):
        mask = Pseudonymiser(salt="fixed")
        assert mask("sync0") != mask("sync1")

    def test_a_different_salt_gives_a_different_pseudonym(self):
        assert Pseudonymiser(salt="a")("sync0") != Pseudonymiser(salt="b")("sync0")

    def test_the_pseudonym_does_not_contain_the_account_id(self):
        assert "sync0" not in Pseudonymiser(salt="fixed")("sync0")

    def test_the_analyst_can_resolve_what_they_issued(self):
        mask = Pseudonymiser(salt="fixed")
        pseudonym = mask("sync0")
        assert mask.resolve(pseudonym) == "sync0"
        assert mask.resolve("acct_neverissued") is None

    def test_disabling_it_is_the_identity(self):
        mask = Pseudonymiser(enabled=False)
        assert mask("sync0") == "sync0"

    def test_a_random_salt_is_used_by_default(self):
        assert Pseudonymiser()("sync0") != Pseudonymiser()("sync0")

    def test_it_counts_what_it_has_seen(self):
        mask = Pseudonymiser(salt="fixed")
        mask("a")
        mask("b")
        mask("a")
        assert len(mask) == 2


# --------------------------------------------------------------------- summary


class TestFeatureSummaries:
    def test_missingness_is_reported_next_to_the_quantiles(self):
        extractor = AccountExtractor()
        frame = extractor.extract(coordinated_corpus())
        summaries = summarise_features(frame, extractor.specs)
        by_name = {s.spec.name: s for s in summaries}
        age = by_name["acct_age_days"]
        assert age.n_measured > 0
        assert age.coverage == pytest.approx(1.0)
        assert age.median is not None

    def test_a_fully_missing_feature_reports_no_quantiles(self):
        # The fixtures carry no follower counts, so this one cannot be computed.
        extractor = AccountExtractor()
        frame = extractor.extract(organic_corpus())
        frame["acct_followback_ratio"] = float("nan")
        summary = next(
            s
            for s in summarise_features(frame, extractor.specs)
            if s.spec.name == "acct_followback_ratio"
        )
        assert summary.n_measured == 0
        assert summary.coverage == 0.0
        assert summary.median is None

    def test_summaries_carry_the_rationale_and_the_limitation(self):
        extractor = AccountExtractor()
        summaries = summarise_features(extractor.extract(organic_corpus()), extractor.specs)
        for summary in summaries:
            exported = summary.as_dict()
            assert exported["rationale"]
            assert exported["limitation"]

    def test_undeclared_columns_are_not_invented(self):
        extractor = AccountExtractor()
        frame = extractor.extract(organic_corpus()).drop(columns=["acct_age_days"])
        names = [s.spec.name for s in summarise_features(frame, extractor.specs)]
        assert "acct_age_days" not in names


class TestClusterCards:
    def test_a_card_describes_the_group_not_its_members(self):
        result = detect_coordination(coordinated_corpus())
        card = build_cards(result.clusters, corpus=coordinated_corpus())[0]
        exported = card.as_dict()
        assert exported["size"] == 4
        assert exported["density"] == 1.0
        # Nothing in the card is keyed by member.
        assert all(not isinstance(value, dict) for value in exported.values())

    def test_members_are_pseudonymous_by_default(self):
        result = detect_coordination(coordinated_corpus())
        card = build_cards(result.clusters, corpus=coordinated_corpus())[0]
        assert PLANTED.isdisjoint(card.members)
        assert all(member.startswith("acct_") for member in card.members)

    def test_real_ids_appear_only_when_explicitly_asked_for(self):
        result = detect_coordination(coordinated_corpus())
        card = build_cards(
            result.clusters,
            corpus=coordinated_corpus(),
            pseudonymiser=Pseudonymiser(enabled=False),
        )[0]
        assert set(card.members) == PLANTED

    def test_every_card_carries_its_caveat(self):
        result = detect_coordination(coordinated_corpus())
        for card in build_cards(result.clusters, corpus=coordinated_corpus()):
            assert "behaviour, not an intent" in card.caveat

    def test_examples_can_be_withheld(self):
        result = detect_coordination(coordinated_corpus())
        card = build_cards(result.clusters, corpus=coordinated_corpus(), include_examples=False)[0]
        assert card.example_post_pairs == ()

    def test_no_clusters_means_no_cards(self):
        result = detect_coordination(organic_corpus())
        assert build_cards(result.clusters, corpus=organic_corpus()) == []


# ---------------------------------------------------------------------- report


class TestReport:
    def test_it_finds_the_planted_cluster(self, report_and_features):
        report, _ = report_and_features
        assert len(report.cards) == 1
        assert report.cards[0].size == 4

    def test_the_feature_matrix_is_returned_not_embedded(self, report_and_features):
        # Per-account values are exactly what a report must not publish, so they
        # come back to the caller instead of going into the document.
        report, features = report_and_features
        assert "sync0" in features.index
        payload = report.as_dict()
        assert all(
            feature["name"].startswith(("acct_", "temp_", "coord_"))
            for feature in payload["features"]
        )
        assert "account_ids" not in json.dumps(payload)

    def test_the_embedded_coordination_result_does_not_relist_the_clusters(
        self, report_and_features
    ):
        # The raw result names each cluster's real accounts. Cards are the one
        # place membership is described, and they are pseudonymised.
        report, _ = report_and_features
        coordination = report.as_dict()["coordination"]
        assert "clusters" not in coordination
        assert coordination["stats"]["n_edges"] == 6
        assert PLANTED.isdisjoint(json.dumps(coordination))

    def test_every_extractor_contributed_features(self, report_and_features):
        report, features = report_and_features
        prefixes = {name.split("_")[0] for name in features.columns}
        assert prefixes == {"acct", "temp", "coord"}
        assert len(report.features) == len(features.columns)

    def test_the_configs_that_shaped_it_travel_with_it(self, report_and_features):
        report, _ = report_and_features
        configs = report.as_dict()["configs"]
        assert configs["coordination"]["window_seconds"] == 900.0
        assert configs["temporal"]["min_posts"] == 20
        assert configs["pseudonymised"] is True

    def test_caveats_are_part_of_the_document(self, report_and_features):
        report, _ = report_and_features
        assert len(report.caveats) >= len(CORPUS_CAVEATS)
        assert any("verdict" in caveat for caveat in report.caveats)

    def test_skipping_the_null_model_is_stated_in_the_report(self, report_and_features):
        report, _ = report_and_features
        assert any("No null model" in caveat for caveat in report.caveats)

    def test_running_the_null_model_removes_that_caveat(self):
        report, _ = build_report(coordinated_corpus(), null_model_permutations=10)
        assert not any("No null model" in caveat for caveat in report.caveats)
        assert report.as_dict()["coordination"]["null_model"]["permutations"] == 10

    def test_disabling_pseudonyms_changes_the_caveat_too(self):
        report, _ = build_report(coordinated_corpus(), pseudonymise=False)
        assert any("real identifiers" in caveat for caveat in report.caveats)

    def test_a_fixed_salt_makes_two_reports_comparable(self):
        first, _ = build_report(coordinated_corpus(), pseudonym_salt="shared")
        second, _ = build_report(coordinated_corpus(), pseudonym_salt="shared")
        assert first.cards[0].members == second.cards[0].members

    def test_the_ingest_report_can_travel_with_the_findings(self, tmp_path: Path):
        path = write_corpus(coordinated_corpus(), tmp_path / "corpus.json")
        loaded = NativeAdapter().load(path)
        report, _ = build_report(loaded.corpus, ingest_report=loaded.report)
        assert report.as_dict()["ingest"]["n_posts"] == len(loaded.corpus)

    def test_an_organic_corpus_reports_nothing_alarming(self):
        report, _ = build_report(organic_corpus())
        assert report.cards == ()
        assert report.as_dict()["coordination"]["stats"]["n_edges"] == 0


class TestExports:
    def test_json_round_trips(self, report_and_features, tmp_path: Path):
        report, _ = report_and_features
        path = tmp_path / "report.json"
        to_json(report, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["title"] == "Test run"
        assert payload["clusters"][0]["size"] == 4
        assert payload["caveats"]

    def test_html_is_self_contained(self, report_and_features, tmp_path: Path):
        report, _ = report_and_features
        path = tmp_path / "report.html"
        html = to_html(report, path)
        assert path.exists()
        assert html.startswith("<!doctype html>")
        # No network request when the file is opened.
        assert "http://" not in html
        assert "<script" not in html
        assert "src=" not in html

    def test_html_leads_with_the_caveats(self, report_and_features):
        report, _ = report_and_features
        html = to_html(report)
        assert html.index("How to read this") < html.index("Feature distributions")

    def test_html_shows_limitations_beside_the_numbers(self, report_and_features):
        report, _ = report_and_features
        html = to_html(report)
        assert "Known limitation" in html
        assert report.features[0].spec.limitation[:40] in html

    def test_html_never_prints_a_real_account_id(self):
        # Post ids stay real -- they are the citation an analyst verifies against
        # -- so this checks a report that withholds them, where nothing else may
        # carry an account identifier.
        report, _ = build_report(coordinated_corpus(), include_examples=False)
        html = to_html(report)
        for account_id in PLANTED:
            assert account_id not in html

    def test_html_escapes_what_it_renders(self):
        report, _ = build_report(coordinated_corpus(), title="<script>alert(1)</script>")
        html = to_html(report)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


# ------------------------------------------------------------------------- cli


class TestCli:
    @pytest.fixture
    def corpus_file(self, tmp_path: Path) -> Path:
        return write_corpus(coordinated_corpus(), tmp_path / "corpus.json")

    def test_analyse_writes_every_requested_output(self, corpus_file: Path, tmp_path: Path):
        html, payload, features = (
            tmp_path / "r.html",
            tmp_path / "r.json",
            tmp_path / "f.csv",
        )
        code = main(
            [
                "analyse",
                str(corpus_file),
                "--html",
                str(html),
                "--json",
                str(payload),
                "--features",
                str(features),
            ]
        )
        assert code == 0
        assert html.exists()
        assert features.exists()
        assert json.loads(payload.read_text(encoding="utf-8"))["clusters"][0]["size"] == 4

    def test_the_window_flag_changes_the_result(self, corpus_file: Path, tmp_path: Path):
        narrow = tmp_path / "narrow.json"
        main(["analyse", str(corpus_file), "--json", str(narrow), "--window", "0.1"])
        assert json.loads(narrow.read_text(encoding="utf-8"))["clusters"] == []

    def test_pseudonyms_are_on_unless_turned_off(self, corpus_file: Path, tmp_path: Path):
        masked, plain = tmp_path / "a.json", tmp_path / "b.json"
        main(["analyse", str(corpus_file), "--json", str(masked)])
        main(["analyse", str(corpus_file), "--json", str(plain), "--no-pseudonyms"])
        members = json.loads(masked.read_text(encoding="utf-8"))["clusters"][0]["members"]
        assert PLANTED.isdisjoint(members)
        plain_members = json.loads(plain.read_text(encoding="utf-8"))["clusters"][0]["members"]
        assert set(plain_members) == PLANTED

    def test_evidence_post_ids_can_be_withheld(self, corpus_file: Path, tmp_path: Path):
        # Post ids stay real by design, so there is a flag for the case where the
        # report travels further than the corpus.
        out = tmp_path / "c.json"
        main(["analyse", str(corpus_file), "--json", str(out), "--no-examples"])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["clusters"][0]["example_post_pairs"] == []
        assert "sync0" not in out.read_text(encoding="utf-8")

    def test_it_reports_what_it_did(self, corpus_file: Path, tmp_path: Path, capsys):
        main(["analyse", str(corpus_file), "--json", str(tmp_path / "out.json")])
        out = capsys.readouterr()
        assert "loaded" in out.out
        assert "cluster" in out.out
        assert "no null model" in out.err

    def test_no_output_flags_says_so(self, corpus_file: Path, capsys):
        assert main(["analyse", str(corpus_file)]) == 0
        assert "no output written" in capsys.readouterr().err

    def test_docs_regenerates_the_catalogue(self, tmp_path: Path):
        out = tmp_path / "features.md"
        assert main(["docs", "--out", str(out)]) == 0
        body = out.read_text(encoding="utf-8")
        assert "# Feature catalogue" in body
        assert "acct_age_days" in body
        assert "temp_circadian_entropy" in body
        assert "coord_partner_count" in body

    def test_an_unknown_command_exits_nonzero(self):
        with pytest.raises(SystemExit) as exit_info:
            main(["frobnicate"])
        assert exit_info.value.code != 0


class TestWindowSweep:
    """The window is the most consequential setting, so it must be sweepable."""

    @pytest.fixture
    def corpus_file(self, tmp_path: Path) -> Path:
        return write_corpus(coordinated_corpus(), tmp_path / "corpus.json")

    def test_the_sweep_is_reported(self, corpus_file: Path, capsys):
        assert main(["analyse", str(corpus_file), "--window-sweep", "0.1", "15", "120"]) == 0
        out = capsys.readouterr().out
        assert "window sensitivity" in out
        # A window too narrow to catch the planted 40-second jitter finds nothing.
        assert "0.1 min  edges       0" in out

    def test_without_the_flag_nothing_is_swept(self, corpus_file: Path, capsys):
        main(["analyse", str(corpus_file)])
        assert "window sensitivity" not in capsys.readouterr().out

    def test_the_candidate_budget_is_reachable_from_the_command_line(self, corpus_file: Path):
        # The safety valve exists precisely so an over-wide window fails fast
        # rather than running for an hour.
        with pytest.raises(ValueError, match="candidate budget"):
            main(["analyse", str(corpus_file), "--max-candidate-pairs", "1"])
