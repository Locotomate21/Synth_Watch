"""Tests for the co-posting graph.

The suite is organised around what the method must find, what it must refuse
to find, and what it is documented to get wrong. The last group matters as
much as the first: a limitation that is not tested is a limitation that
quietly stops being true.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from itertools import combinations
from typing import ClassVar, cast

import pytest

from synthwatch.detect.coordination import (
    BANDS,
    Cluster,
    CoordinationConfig,
    CoordinationExtractor,
    build_graph,
    detect_coordination,
    hamming64,
    normalise_text,
    permutation_test,
    simhash64,
    summarise_clusters,
    window_sensitivity,
)
from synthwatch.models import build_corpus
from synthwatch.types import PostKind
from tests.conftest import make_corpus, make_post
from tests.detect.fixtures_coordination import (
    ORGANIC_TEXTS,
    TEMPLATES,
    TWEAKS,
    coordinated_corpus,
    coordinated_posts,
    headline_corpus,
    organic_corpus,
    organic_posts,
    thread_corpus,
)

PLANTED = {"sync0", "sync1", "sync2", "sync3"}


# ---------------------------------------------------------------- fingerprint


class TestNormalisation:
    def test_urls_are_stripped(self):
        base = TEMPLATES[0]
        assert simhash64(base) == simhash64(f"{base} https://example.org/a?utm_source=x")

    def test_mentions_are_stripped(self):
        base = TEMPLATES[0]
        assert simhash64(f"@alguien {base}") == simhash64(f"@otra_persona {base}")

    def test_case_and_width_fold(self):
        assert normalise_text("ABC  def") == normalise_text("abc def")
        assert normalise_text("ｈｏｌａ") == "hola"  # noqa: RUF001

    def test_empty_text_has_a_zero_fingerprint(self):
        assert simhash64("") == 0


class TestSimhash:
    def test_identical_text_has_distance_zero(self):
        assert hamming64(simhash64(TEMPLATES[0]), simhash64(TEMPLATES[0])) == 0

    def test_cosmetic_edits_stay_within_the_default_threshold(self):
        # This is the calibration the default hamming_threshold rests on.
        threshold = CoordinationConfig().hamming_threshold
        prints = [simhash64(TEMPLATES[0] + tweak) for tweak in TWEAKS]
        worst = max(hamming64(a, b) for a, b in combinations(prints, 2))
        assert worst <= threshold

    def test_unrelated_text_is_far_away(self):
        prints = [simhash64(text) for text in ORGANIC_TEXTS]
        closest = min(hamming64(a, b) for a, b in combinations(prints, 2))
        assert closest > 2 * CoordinationConfig().hamming_threshold

    def test_fingerprints_survive_a_different_hash_seed(self):
        # PYTHONHASHSEED randomises str.__hash__; using it here would make every
        # run produce different fingerprints and silently break reproducibility.
        script = (
            "from synthwatch.detect.coordination import simhash64;"
            "print(simhash64('la reforma que aprobaron de madrugada no arregla nada'))"
        )
        outputs = set()
        for seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
            )
            outputs.add(result.stdout.strip())
        assert len(outputs) == 1


# --------------------------------------------------------------------- config


class TestConfig:
    def test_window_must_be_positive(self):
        with pytest.raises(ValueError, match="window must be positive"):
            CoordinationConfig(window=timedelta(0))

    def test_threshold_above_the_band_count_is_refused(self):
        # Above this, band indexing stops being exhaustive and pairs would be
        # dropped without anyone noticing.
        with pytest.raises(ValueError, match="hamming_threshold"):
            CoordinationConfig(hamming_threshold=BANDS)

    def test_min_edge_weight_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="min_edge_weight"):
            CoordinationConfig(min_edge_weight=0)

    def test_config_is_serialisable_for_the_report(self):
        assert CoordinationConfig().as_dict()["window_seconds"] == 900.0


# ---------------------------------------------------------------------- graph


class TestGraph:
    def test_planted_cluster_is_fully_connected(self):
        graph, _ = build_graph(coordinated_corpus())
        for left, right in combinations(sorted(PLANTED), 2):
            assert graph.has_edge(left, right), f"missing {left}-{right}"

    def test_organic_accounts_stay_isolated(self):
        graph, _ = build_graph(coordinated_corpus())
        organic = [node for node in graph if node.startswith("org")]
        assert organic
        assert all(graph.degree(node) == 0 for node in organic)

    def test_edges_carry_traceable_evidence(self):
        graph, _ = build_graph(coordinated_corpus())
        data = graph["sync0"]["sync1"]
        assert data["weight"] == 3  # one per wave
        assert data["mean_similarity"] > 0.9
        assert data["median_lag_seconds"] == pytest.approx(40.0)
        assert data["examples"], "an edge must point back at the posts behind it"

    def test_nothing_is_found_in_an_organic_corpus(self):
        graph, stats = build_graph(organic_corpus())
        assert graph.number_of_edges() == 0
        assert stats["n_posts_matched"] == 0

    def test_single_coincidence_does_not_make_an_edge(self):
        # One shared template is a coincidence; min_edge_weight is what keeps it
        # from being reported as a coordinated pair.
        corpus = make_corpus(coordinated_posts(n_waves=1))
        graph, _ = build_graph(corpus)
        assert graph.number_of_edges() == 0

    def test_same_content_outside_the_window_does_not_match(self):
        corpus = make_corpus(coordinated_posts(jitter_seconds=60 * 60))
        graph, _ = build_graph(corpus, CoordinationConfig(window=timedelta(minutes=15)))
        assert graph.number_of_edges() == 0

    def test_short_posts_are_excluded(self):
        corpus = make_corpus(
            [
                make_post("p1", "a1", offset_minutes=0, text="totalmente de acuerdo"),
                make_post("p2", "a2", offset_minutes=1, text="totalmente de acuerdo"),
                make_post("p3", "a1", offset_minutes=60, text="totalmente de acuerdo"),
                make_post("p4", "a2", offset_minutes=61, text="totalmente de acuerdo"),
            ]
        )
        graph, stats = build_graph(corpus)
        assert stats["n_posts_eligible"] == 0
        assert graph.number_of_edges() == 0

    def test_reposts_are_excluded_by_default_and_countable_on_request(self):
        posts = [
            make_post(
                f"r{account}_{wave}",
                f"acct{account}",
                offset_minutes=wave * 240 + account,
                text=TEMPLATES[wave % len(TEMPLATES)],
                kind=PostKind.REPOST,
            )
            for wave in range(3)
            for account in range(3)
        ]
        corpus = make_corpus(posts)
        assert build_graph(corpus)[0].number_of_edges() == 0
        with_reposts = build_graph(corpus, CoordinationConfig(include_reposts=True))[0]
        assert with_reposts.number_of_edges() == 3

    def test_analysis_is_deterministic(self):
        corpus = coordinated_corpus()
        first, _ = build_graph(corpus)
        second, _ = build_graph(corpus)
        assert sorted(first.edges()) == sorted(second.edges())
        assert [c.as_dict() for c in detect_coordination(corpus).clusters] == [
            c.as_dict() for c in detect_coordination(corpus).clusters
        ]

    def test_candidate_budget_is_enforced(self):
        corpus = coordinated_corpus()
        with pytest.raises(ValueError, match="candidate budget"):
            build_graph(corpus, CoordinationConfig(max_candidate_pairs=1))


class TestThreadGuard:
    def test_one_busy_thread_is_not_coordination(self):
        graph, _ = build_graph(thread_corpus())
        assert graph.number_of_edges() == 0

    def test_the_guard_is_what_suppresses_it(self):
        # Same data, thread awareness off: the conversation turns into a dense
        # cluster. This is what a source without thread ids would produce.
        graph, _ = build_graph(thread_corpus(), CoordinationConfig(exclude_same_thread=False))
        assert graph.number_of_edges() > 0

    def test_replies_in_different_threads_still_count(self):
        posts = [
            make_post(
                f"x{account}_{wave}",
                f"acct{account}",
                offset_minutes=wave * 240 + account,
                text=TEMPLATES[wave % len(TEMPLATES)],
                kind=PostKind.REPLY,
                root_post_id=f"root_{account}",
            )
            for wave in range(3)
            for account in range(3)
        ]
        graph, _ = build_graph(make_corpus(posts))
        assert graph.number_of_edges() == 3


# ------------------------------------------------------------------- clusters


class TestClusters:
    def test_the_planted_cluster_is_recovered(self):
        result = detect_coordination(coordinated_corpus())
        assert len(result.clusters) == 1
        assert set(result.clusters[0].account_ids) == PLANTED

    def test_a_complete_cluster_has_density_one(self):
        cluster = detect_coordination(coordinated_corpus()).clusters[0]
        assert cluster.density == pytest.approx(1.0)
        assert cluster.size == 4
        assert cluster.total_pairs == 18  # 6 pairs x 3 waves

    def test_small_communities_are_not_reported(self):
        config = CoordinationConfig(min_cluster_size=5)
        assert detect_coordination(coordinated_corpus(), config).clusters == ()

    def test_organic_corpus_yields_no_clusters(self):
        result = detect_coordination(organic_corpus())
        assert result.clusters == ()
        assert result.clustered_account_ids == frozenset()

    def test_cluster_lookup(self):
        result = detect_coordination(coordinated_corpus())
        assert result.cluster_of("sync0") is result.clusters[0]
        assert result.cluster_of("org0") is None

    def test_export_carries_the_caveat(self):
        exported = detect_coordination(coordinated_corpus()).as_dict()
        config = cast("dict[str, object]", exported["config"])
        clusters = cast("list[dict[str, object]]", exported["clusters"])
        assert "behaviour, not an" in str(exported["caveat"])
        assert config["window_seconds"] == 900.0
        assert clusters[0]["size"] == 4

    def test_summary_of_clusters(self):
        clusters = detect_coordination(coordinated_corpus()).clusters
        assert summarise_clusters(clusters) == {
            "n_clusters": 1,
            "n_accounts_in_clusters": 4,
            "largest_cluster": 4,
            "median_cluster_size": 4,
        }

    def test_summary_of_nothing(self):
        assert summarise_clusters([])["n_clusters"] == 0


class TestDocumentedFalsePositives:
    def test_a_shared_headline_looks_exactly_like_coordination(self):
        # The method cannot tell these apart, and pretending otherwise would be
        # the real failure. The test pins the limitation so it stays documented.
        result = detect_coordination(headline_corpus())
        assert len(result.clusters) == 1
        assert result.clusters[0].size == 4


# ----------------------------------------------------------------- null model


class TestNullModel:
    def test_planted_coordination_beats_chance(self):
        result = permutation_test(coordinated_corpus(), permutations=30)
        assert result.observed == 6
        assert result.p_value < 0.05
        assert max(result.permuted) < result.observed

    def test_an_organic_corpus_cannot_reject_the_null(self):
        result = permutation_test(organic_corpus(), permutations=10)
        assert result.observed == 0
        assert result.p_value == 1.0

    def test_p_value_is_never_zero(self):
        result = permutation_test(coordinated_corpus(), permutations=5)
        assert result.p_value >= 1 / 6

    def test_shuffle_strategy_is_available(self):
        result = permutation_test(coordinated_corpus(), permutations=10, strategy="shuffle")
        assert result.strategy == "shuffle"
        assert result.p_value < 0.2

    def test_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match="unknown null model strategy"):
            permutation_test(organic_corpus(), permutations=1, strategy="jitter")

    def test_result_is_serialisable(self):
        exported = permutation_test(coordinated_corpus(), permutations=10).as_dict()
        assert exported["observed_edges"] == 6
        assert exported["permutations"] == 10

    def test_detect_can_run_the_test_inline(self):
        result = detect_coordination(coordinated_corpus(), null_model_permutations=10)
        assert result.null_model is not None
        null_model = cast("dict[str, object]", result.as_dict()["null_model"])
        assert null_model["strategy"] == "shift"

    def test_null_model_is_skipped_by_default(self):
        assert detect_coordination(organic_corpus()).null_model is None


# -------------------------------------------------------------------- features


class TestExtractor:
    def test_every_feature_is_documented_and_prefixed(self):
        for spec in CoordinationExtractor().specs:
            assert spec.name.startswith("coord_")
            assert len(spec.rationale) >= 40
            assert len(spec.limitation) >= 40

    def test_frame_covers_every_account(self):
        corpus = coordinated_corpus()
        frame = CoordinationExtractor().extract(corpus)
        assert set(frame.index) == {a.account_id for a in corpus.accounts}
        assert list(frame.columns) == [s.name for s in CoordinationExtractor().specs]

    def test_planted_accounts_score_as_expected(self):
        frame = CoordinationExtractor().extract(coordinated_corpus())
        row = frame.loc["sync0"]
        assert row["coord_partner_count"] == 3
        assert row["coord_pair_count"] == 9
        assert row["coord_max_similarity"] > 0.9
        assert row["coord_median_lag_seconds"] == pytest.approx(80.0)
        assert row["coord_duplicate_post_ratio"] == 1.0
        assert row["coord_cluster_size"] == 4
        assert row["coord_cluster_density"] == pytest.approx(1.0)

    def test_analysed_but_unmatched_accounts_score_zero_not_nan(self):
        row = CoordinationExtractor().extract(coordinated_corpus()).loc["org0"]
        assert row["coord_partner_count"] == 0
        assert row["coord_duplicate_post_ratio"] == 0
        assert row["coord_cluster_size"] == 1
        # No partner means no lag to measure: that stays NaN rather than 0.
        assert row["coord_median_lag_seconds"] != row["coord_median_lag_seconds"]

    def test_accounts_with_nothing_analysable_stay_nan(self):
        # "We could not look" must never be recorded as "we looked and found
        # nothing" -- the ensemble needs to tell those two apart.
        corpus = make_corpus(
            [
                *coordinated_posts(),
                make_post("tiny1", "quiet", offset_minutes=5, text="jaja si"),
            ]
        )
        row = CoordinationExtractor().extract(corpus).loc["quiet"]
        assert row.isna().all()

    def test_thread_participants_are_not_scored_as_partners(self):
        frame = CoordinationExtractor().extract(thread_corpus())
        assert frame.loc["thr0", "coord_partner_count"] == 0
        assert frame.loc["thr0", "coord_duplicate_post_ratio"] == 0


# --------------------------------------------------------------- sensitivity


def test_window_sensitivity_shows_the_parameter_driving_the_result():
    rows = window_sensitivity(
        coordinated_corpus(),
        [timedelta(seconds=10), timedelta(minutes=15), timedelta(hours=2)],
    )
    assert [row["window_seconds"] for row in rows] == [10.0, 900.0, 7200.0]
    assert rows[0]["n_edges"] == 0  # too narrow to catch the 40s jitter
    assert rows[1]["n_edges"] == 6
    assert rows[2]["n_accounts_in_clusters"] == 4


def test_orphan_authors_do_not_break_extraction():
    # A post whose author was never crawled: common in real collections.
    corpus = build_corpus([], [*coordinated_posts(), *organic_posts()])
    frame = CoordinationExtractor().extract(corpus)
    assert frame.loc["sync0", "coord_partner_count"] == 3


def test_cluster_is_immutable():
    cluster = detect_coordination(coordinated_corpus()).clusters[0]
    assert isinstance(cluster, Cluster)
    with pytest.raises((AttributeError, TypeError)):
        cluster.density = 0.0  # type: ignore[misc]


class TestFingerprintStability:
    """The fingerprint is a stored format, not an implementation detail.

    Corpora are cached, reports are compared across runs, and a near-duplicate
    threshold is calibrated against a particular distance distribution. Change
    the fingerprint and all of that silently stops meaning what it meant.
    """

    GOLDEN: ClassVar[dict[str, int]] = {
        "hola que tal": 312879931155061397,
        "La reforma que aprobaron de madrugada no arregla nada": 1222999411927696821,
    }

    @pytest.mark.parametrize(("text", "expected"), list(GOLDEN.items()))
    def test_known_texts_keep_their_fingerprints(self, text: str, expected: int):
        assert simhash64(text) == expected

    def test_the_shingle_size_is_part_of_the_format(self):
        # Two sizes are two different fingerprint spaces; mixing them silently
        # would make every distance meaningless.
        assert simhash64("hola que tal", shingle_size=4) != self.GOLDEN["hola que tal"]

    def test_repeated_shingles_are_weighted_not_deduplicated(self):
        # "aaaa" has one distinct shingle repeated; "abcd" has four distinct
        # ones. If multiplicity were dropped, the first would match anything
        # else built from a single repeated shingle.
        assert simhash64("a" * 40) != simhash64("b" * 40)
        assert simhash64("abab" * 10) != simhash64("ab" * 20 + "cd")
