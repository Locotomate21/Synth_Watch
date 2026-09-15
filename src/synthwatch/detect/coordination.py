"""Co-posting graph: who publishes near-identical content, near-simultaneously.

The method, in order:

1. **Fingerprint.** Every eligible post is reduced to a 64-bit SimHash over
   character shingles of its normalised text (:func:`simhash64`).
2. **Candidate generation.** Fingerprints are indexed by 8-bit bands crossed
   with time buckets the width of the window. Because a pair differing in
   ``d`` bits leaves at least ``8 - d`` bands identical, banding with 8 bands
   is *exact* for any threshold below 8: it discards no true pair, it only
   avoids comparing everything with everything.
3. **Edges.** A candidate pair becomes evidence when the two posts come from
   different accounts, fall within the window, and differ by no more than
   ``hamming_threshold`` bits. Evidence accumulates on the account pair; an
   edge is drawn once it reaches ``min_edge_weight`` distinct co-posts.
4. **Communities.** Louvain over the weighted graph, with a fixed seed so a
   report can be reproduced exactly.
5. **Null model.** The observed graph is compared against corpora where the
   timing structure has been destroyed but the content has not
   (:func:`permutation_test`), because "50 accounts posted similar things" is
   meaningless without knowing what chance produces in the same corpus.

What this measures, and what it does not
----------------------------------------

This finds *synchronised near-duplicate publishing*. That is a behaviour, not
an intent and not an identity. Campaign volunteers working from a shared
message guide, fandoms, newsrooms syndicating a wire story, and people
reacting to the same push notification all generate exactly this signature.
A dense cluster is a question worth investigating; on its own it is not a
finding, and the report layer is required to carry that caveat with it.

Known limitations
-----------------

* **Short text is noise.** Below ``min_chars`` normalised characters, SimHash
  collisions stop being informative, so short posts are excluded outright.
  That silently removes most of the reply-guy and one-word-support corpus.
* **Reposts are excluded by default.** Verbatim amplification is a platform
  feature, and including it would flood the graph with ordinary sharing.
* **A single thread is not coordination.** Pairs sharing ``root_post_id`` are
  ignored, which is only as good as the adapter that filled that field in. A
  source without thread identifiers will over-detect busy conversations.
* **Translation and paraphrase are invisible.** SimHash is lexical: the same
  talking point rewritten by a language model produces no edge at all. The
  method finds copy-paste, so it will systematically under-report the more
  capable operations.
* **The window is the finding.** Widening the window from 5 to 60 minutes can
  turn a null result into a dense graph; the value used must be reported.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from statistics import median

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities

from synthwatch.detect.base import FeatureSpec, empty_frame
from synthwatch.models import Corpus, Post
from synthwatch.types import AccountId, ClusterId, FeatureLevel, PostId, PostKind

__all__ = [
    "Cluster",
    "CoordinationConfig",
    "CoordinationExtractor",
    "CoordinationResult",
    "NullModelResult",
    "build_graph",
    "detect_coordination",
    "hamming64",
    "normalise_text",
    "permutation_test",
    "simhash64",
]

HASH_BITS = 64
BANDS = 8
"""Banding is exact for any Hamming threshold below this (pigeonhole)."""

BAND_BITS = HASH_BITS // BANDS
BAND_MASK = (1 << BAND_BITS) - 1

MAX_EDGE_EXAMPLES = 3
"""Post pairs kept per edge, so a report can trace evidence without hoarding text."""

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# NFKC folding runs first, so a full-width mention is already plain "@" here.
_MENTION_RE = re.compile(r"@[\w.\-]+")
_WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """Reduce ``text`` to the form the fingerprint is computed over.

    Applies NFKC folding (so full-width and decomposed variants collapse),
    case folding, removal of URLs and @-mentions, and whitespace collapsing.

    URLs and mentions are dropped because they are the two things that differ
    between otherwise identical copies of the same message: tracking
    parameters are rewritten per share, and a mention changes with whoever is
    being answered. Dropping them is what lets a copy-paste campaign stay
    visible. It also means a set of accounts pushing the *same link* with
    different wording produces no edge here; that signal belongs to a co-link
    graph, which this module does not build.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    without_urls = _URL_RE.sub(" ", folded)
    without_mentions = _MENTION_RE.sub(" ", without_urls)
    return _WHITESPACE_RE.sub(" ", without_mentions).strip()


def _shingles(text: str, size: int) -> Counter[str]:
    """Character n-grams of ``text`` with their multiplicities."""
    if len(text) <= size:
        return Counter([text]) if text else Counter()
    return Counter(text[i : i + size] for i in range(len(text) - size + 1))


def simhash64(text: str, *, shingle_size: int = 5) -> int:
    """Compute the 64-bit SimHash of ``text`` after :func:`normalise_text`.

    Character shingles are used rather than words so that the fingerprint
    survives the small edits -- an emoji swapped, a word reordered, a hashtag
    appended -- that template-driven posting introduces to defeat exact-match
    deduplication.

    Uses BLAKE2b rather than :func:`hash`, whose randomised seed would make
    every run of the pipeline produce different fingerprints.
    """
    counts = _shingles(normalise_text(text), shingle_size)
    if not counts:
        return 0
    vector = [0] * HASH_BITS
    for shingle, weight in counts.items():
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(HASH_BITS):
            vector[bit] += weight if (value >> bit) & 1 else -weight
    return sum(1 << bit for bit in range(HASH_BITS) if vector[bit] > 0)


def hamming64(left: int, right: int) -> int:
    """Number of differing bits between two 64-bit fingerprints."""
    return ((left ^ right) & ((1 << HASH_BITS) - 1)).bit_count()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoordinationConfig:
    """Parameters of the co-posting analysis.

    Every default here is a modelling choice, not a fact about the world, and
    every one of them changes the result. They belong in the report.

    Attributes:
        window: Maximum time between two posts for them to count as
            co-published. Short windows find automation, long ones find
            audiences reacting to the same event.
        hamming_threshold: Maximum fingerprint distance, in bits, out of 64.
            The classic value for 64-bit SimHash is 3, calibrated on web pages;
            social posts are two orders of magnitude shorter, where a single
            edited character moves two to four bits, so the default here is 6
            (about 90% bit agreement). Unrelated posts sit near 32 bits, so the
            margin remains wide. Must stay below 8 for candidate generation to
            remain exact.
        shingle_size: Character n-gram size for the fingerprint.
        min_chars: Posts shorter than this after normalisation are dropped,
            because SimHash on short strings collides for uninteresting
            reasons.
        include_reposts: Whether verbatim reshares count as co-publishing.
        exclude_same_thread: Whether to ignore pairs sharing a conversation
            root, so one busy thread is not read as a coordinated campaign.
        min_edge_weight: Co-posts required before an account pair gets an
            edge. At 1, a single shared meme creates a "coordinated pair".
        min_cluster_size: Smallest community reported as a cluster.
        resolution: Louvain resolution; above 1 yields smaller communities.
        seed: Fixes Louvain and the null model, so results are reproducible.
        max_candidate_pairs: Safety valve. Exceeding it raises rather than
            spending an hour on an over-wide window.
    """

    window: timedelta = timedelta(minutes=15)
    hamming_threshold: int = 6
    shingle_size: int = 5
    min_chars: int = 30
    include_reposts: bool = False
    exclude_same_thread: bool = True
    min_edge_weight: int = 2
    min_cluster_size: int = 3
    resolution: float = 1.0
    seed: int = 20240301
    max_candidate_pairs: int = 5_000_000

    def __post_init__(self) -> None:
        """Reject parameter combinations that would silently misbehave."""
        if self.window <= timedelta(0):
            msg = "window must be positive"
            raise ValueError(msg)
        if not 0 <= self.hamming_threshold < BANDS:
            msg = (
                f"hamming_threshold must be in [0, {BANDS}); above that, band "
                "indexing stops being exact and pairs would be missed silently"
            )
            raise ValueError(msg)
        if self.shingle_size < 1:
            msg = "shingle_size must be >= 1"
            raise ValueError(msg)
        if self.min_edge_weight < 1:
            msg = "min_edge_weight must be >= 1"
            raise ValueError(msg)

    @property
    def window_seconds(self) -> float:
        """The window expressed in seconds."""
        return self.window.total_seconds()

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, meant to be embedded in every report."""
        return {
            "window_seconds": self.window_seconds,
            "hamming_threshold": self.hamming_threshold,
            "shingle_size": self.shingle_size,
            "min_chars": self.min_chars,
            "include_reposts": self.include_reposts,
            "exclude_same_thread": self.exclude_same_thread,
            "min_edge_weight": self.min_edge_weight,
            "min_cluster_size": self.min_cluster_size,
            "resolution": self.resolution,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------
# Internal representation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    """One eligible post, reduced to what the graph builder needs."""

    post_id: PostId
    account_id: AccountId
    timestamp: float
    fingerprint: int
    root_id: str | None
    text: str


@dataclass(slots=True)
class _PairEvidence:
    """Accumulated co-posting evidence for one account pair."""

    similarities: list[float] = field(default_factory=list)
    lags: list[float] = field(default_factory=list)
    examples: list[tuple[PostId, PostId]] = field(default_factory=list)


def _eligible(post: Post, config: CoordinationConfig) -> bool:
    """Whether a post can contribute evidence at all."""
    if not config.include_reposts and post.kind is PostKind.REPOST:
        return False
    return len(normalise_text(post.text)) >= config.min_chars


def _fingerprints(corpus: Corpus, config: CoordinationConfig) -> list[_Fingerprint]:
    """Fingerprint every eligible post, in canonical chronological order."""
    return [
        _Fingerprint(
            post_id=post.post_id,
            account_id=post.account_id,
            timestamp=post.created_at.timestamp(),
            fingerprint=simhash64(post.text, shingle_size=config.shingle_size),
            root_id=post.root_post_id,
            text=post.text,
        )
        for post in corpus.posts
        if _eligible(post, config)
    ]


def _accumulate_pairs(
    records: Sequence[_Fingerprint], config: CoordinationConfig
) -> tuple[dict[tuple[AccountId, AccountId], _PairEvidence], set[PostId], int]:
    """Compare banded, time-bucketed candidates and accumulate pair evidence.

    Returns:
        The evidence per account pair, the ids of posts that matched something,
        and the number of candidate comparisons performed.

    Raises:
        ValueError: If the candidate budget in the config is exhausted.
    """
    window = config.window_seconds
    ordered = sorted(records, key=lambda r: (r.timestamp, r.post_id))
    index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    evidence: dict[tuple[AccountId, AccountId], _PairEvidence] = {}
    matched: set[PostId] = set()
    comparisons = 0

    for position, record in enumerate(ordered):
        bucket = int(record.timestamp // window)
        candidates: set[int] = set()
        for band in range(BANDS):
            value = (record.fingerprint >> (band * BAND_BITS)) & BAND_MASK
            # A pair within the window sits in the same bucket or the previous
            # one, so these two lookups are exhaustive.
            for neighbour in (bucket, bucket - 1):
                candidates.update(index[(band, value, neighbour)])
            index[(band, value, bucket)].append(position)

        for other in candidates:
            previous = ordered[other]
            comparisons += 1
            if comparisons > config.max_candidate_pairs:
                msg = (
                    f"candidate budget of {config.max_candidate_pairs} exhausted; "
                    "narrow the window, raise min_chars, or analyse a subset"
                )
                raise ValueError(msg)
            if previous.account_id == record.account_id:
                continue
            lag = record.timestamp - previous.timestamp
            if lag > window:
                continue
            if (
                config.exclude_same_thread
                and previous.root_id is not None
                and previous.root_id == record.root_id
            ):
                continue
            distance = hamming64(previous.fingerprint, record.fingerprint)
            if distance > config.hamming_threshold:
                continue

            key = (
                min(previous.account_id, record.account_id),
                max(previous.account_id, record.account_id),
            )
            entry = evidence.setdefault(key, _PairEvidence())
            entry.similarities.append(1.0 - distance / HASH_BITS)
            entry.lags.append(abs(lag))
            if len(entry.examples) < MAX_EDGE_EXAMPLES:
                entry.examples.append((previous.post_id, record.post_id))
            matched.add(previous.post_id)
            matched.add(record.post_id)

    return evidence, matched, comparisons


# --------------------------------------------------------------------------
# Graph and clusters
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Analysis:
    """Everything one pass over the corpus produces, computed once."""

    graph: nx.Graph
    stats: dict[str, object]
    matched: frozenset[PostId]
    records: tuple[_Fingerprint, ...]


def _analyse(corpus: Corpus, config: CoordinationConfig) -> _Analysis:
    """Fingerprint, pair up and build the graph in a single pass."""
    records = _fingerprints(corpus, config)
    evidence, matched, comparisons = _accumulate_pairs(records, config)

    graph = nx.Graph()
    graph.add_nodes_from(sorted({record.account_id for record in records}))
    for (left, right), entry in evidence.items():
        weight = len(entry.similarities)
        if weight < config.min_edge_weight:
            continue
        graph.add_edge(
            left,
            right,
            weight=weight,
            mean_similarity=sum(entry.similarities) / weight,
            median_lag_seconds=median(entry.lags),
            examples=tuple(entry.examples),
        )

    stats: dict[str, object] = {
        "n_posts_total": len(corpus.posts),
        "n_posts_eligible": len(records),
        "n_posts_matched": len(matched),
        "n_candidate_comparisons": comparisons,
        "n_pairs_with_evidence": len(evidence),
        "n_edges": graph.number_of_edges(),
        "n_nodes": graph.number_of_nodes(),
    }
    return _Analysis(graph, stats, frozenset(matched), tuple(records))


def build_graph(
    corpus: Corpus, config: CoordinationConfig | None = None
) -> tuple[nx.Graph, dict[str, object]]:
    """Build the weighted co-posting graph for ``corpus``.

    Nodes are account ids; an edge carries ``weight`` (number of co-published
    near-duplicate pairs), ``mean_similarity``, ``median_lag_seconds`` and up
    to three ``examples`` of the post pairs behind it, so that any edge in a
    report can be traced back to the posts that produced it.

    Returns:
        The graph and a stats dictionary describing what was filtered out.
    """
    analysis = _analyse(corpus, config or CoordinationConfig())
    return analysis.graph, analysis.stats


@dataclass(frozen=True, slots=True)
class Cluster:
    """A community of accounts publishing near-duplicate content in sync.

    This is a description of observed behaviour. It is deliberately not called
    a network, a campaign or a botnet, and it carries no attribution.
    """

    cluster_id: ClusterId
    account_ids: tuple[AccountId, ...]
    n_edges: int
    total_pairs: int
    density: float
    mean_similarity: float
    median_lag_seconds: float
    examples: tuple[tuple[PostId, PostId], ...]

    @property
    def size(self) -> int:
        """Number of accounts in the cluster."""
        return len(self.account_ids)

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the report layer."""
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "account_ids": list(self.account_ids),
            "n_edges": self.n_edges,
            "total_pairs": self.total_pairs,
            "density": round(self.density, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "median_lag_seconds": round(self.median_lag_seconds, 1),
            "examples": [list(pair) for pair in self.examples],
        }


@dataclass(frozen=True, slots=True)
class NullModelResult:
    """Observed edge count against what the same corpus produces by chance."""

    observed: int
    permuted: tuple[int, ...]
    p_value: float
    strategy: str

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the report layer."""
        permuted_mean = sum(self.permuted) / len(self.permuted) if self.permuted else 0.0
        return {
            "strategy": self.strategy,
            "observed_edges": self.observed,
            "permutations": len(self.permuted),
            "permuted_mean_edges": round(permuted_mean, 2),
            "p_value": round(self.p_value, 4),
        }


@dataclass(frozen=True, slots=True)
class CoordinationResult:
    """Everything the coordination analysis produced, plus how it was produced."""

    graph: nx.Graph
    clusters: tuple[Cluster, ...]
    config: CoordinationConfig
    stats: Mapping[str, object]
    null_model: NullModelResult | None = None

    @property
    def clustered_account_ids(self) -> frozenset[AccountId]:
        """Accounts belonging to a reported cluster."""
        return frozenset(a for cluster in self.clusters for a in cluster.account_ids)

    def cluster_of(self, account_id: AccountId) -> Cluster | None:
        """The cluster containing ``account_id``, if any."""
        return next((c for c in self.clusters if account_id in c.account_ids), None)

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, with the caveat attached to the numbers."""
        return {
            "config": self.config.as_dict(),
            "stats": dict(self.stats),
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "null_model": self.null_model.as_dict() if self.null_model else None,
            "caveat": (
                "Synchronised near-duplicate publishing is a behaviour, not an "
                "intent. Shared message guides, fandoms, syndicated wire copy and "
                "scheduling tools produce the same signature."
            ),
        }


def _clusters_from_graph(graph: nx.Graph, config: CoordinationConfig) -> tuple[Cluster, ...]:
    """Run Louvain and summarise the communities worth reporting."""
    if graph.number_of_edges() == 0:
        return ()
    communities = louvain_communities(
        graph, weight="weight", resolution=config.resolution, seed=config.seed
    )
    clusters: list[Cluster] = []
    # Sorted by size then by first account id, so cluster_ids are stable across
    # runs regardless of the order Louvain happens to return communities in.
    ordered = sorted(communities, key=lambda c: (-len(c), min(c)))
    for position, community in enumerate(ordered):
        if len(community) < config.min_cluster_size:
            continue
        subgraph = graph.subgraph(community)
        edges = list(subgraph.edges(data=True))
        if not edges:
            continue
        size = len(community)
        possible = size * (size - 1) / 2
        total_pairs = sum(int(data["weight"]) for _, _, data in edges)
        clusters.append(
            Cluster(
                cluster_id=f"c{position:03d}",
                account_ids=tuple(sorted(community)),
                n_edges=len(edges),
                total_pairs=total_pairs,
                density=len(edges) / possible if possible else 0.0,
                mean_similarity=sum(float(d["mean_similarity"]) for _, _, d in edges) / len(edges),
                median_lag_seconds=median([float(d["median_lag_seconds"]) for _, _, d in edges]),
                examples=tuple(pair for _, _, d in edges[:3] for pair in tuple(d["examples"])[:1]),
            )
        )
    return tuple(clusters)


def detect_coordination(
    corpus: Corpus,
    config: CoordinationConfig | None = None,
    *,
    null_model_permutations: int = 0,
    null_model_strategy: str = "shift",
) -> CoordinationResult:
    """Run the full coordination analysis over ``corpus``.

    Args:
        corpus: The corpus to analyse.
        config: Analysis parameters; defaults are documented on
            :class:`CoordinationConfig` and belong in the report.
        null_model_permutations: Permutations for the significance check. Zero
            skips it -- which means the resulting cluster count has nothing to
            be compared against, so prefer at least 50 for anything published.
        null_model_strategy: ``"shift"`` (each account keeps its own rhythm,
            its timeline slides) or ``"shuffle"`` (timestamps swapped between
            posts). ``"shift"`` is the more conservative of the two.
    """
    config = config or CoordinationConfig()
    analysis = _analyse(corpus, config)
    graph, stats = analysis.graph, analysis.stats
    clusters = _clusters_from_graph(graph, config)
    null_model = (
        permutation_test(
            corpus,
            config,
            permutations=null_model_permutations,
            strategy=null_model_strategy,
        )
        if null_model_permutations > 0
        else None
    )
    return CoordinationResult(
        graph=graph,
        clusters=clusters,
        config=config,
        stats={**stats, "n_clusters": len(clusters)},
        null_model=null_model,
    )


# --------------------------------------------------------------------------
# Null model
# --------------------------------------------------------------------------


def _shifted(records: Sequence[_Fingerprint], rng: random.Random) -> list[_Fingerprint]:
    """Slide each account's whole timeline by an independent random offset.

    Preserves every account's internal rhythm -- burstiness, circadian shape,
    inter-arrival distribution -- and destroys only the alignment *between*
    accounts, which is the thing being measured.
    """
    if not records:
        return []
    span = max(r.timestamp for r in records) - min(r.timestamp for r in records)
    offsets: dict[AccountId, float] = {}
    for record in records:
        if record.account_id not in offsets:
            offsets[record.account_id] = rng.uniform(-span / 2, span / 2) if span else 0.0
    return [
        _Fingerprint(
            post_id=r.post_id,
            account_id=r.account_id,
            timestamp=r.timestamp + offsets[r.account_id],
            fingerprint=r.fingerprint,
            root_id=r.root_id,
            text=r.text,
        )
        for r in records
    ]


def _shuffled(records: Sequence[_Fingerprint], rng: random.Random) -> list[_Fingerprint]:
    """Swap timestamps between posts, keeping the corpus-wide time profile."""
    timestamps = [r.timestamp for r in records]
    rng.shuffle(timestamps)
    return [
        _Fingerprint(
            post_id=r.post_id,
            account_id=r.account_id,
            timestamp=stamp,
            fingerprint=r.fingerprint,
            root_id=r.root_id,
            text=r.text,
        )
        for r, stamp in zip(records, timestamps, strict=True)
    ]


def permutation_test(
    corpus: Corpus,
    config: CoordinationConfig | None = None,
    *,
    permutations: int = 100,
    strategy: str = "shift",
) -> NullModelResult:
    """Compare the observed edge count with time-randomised versions of itself.

    Content is held fixed and only timing is randomised, so the test answers
    one narrow question: is the *synchronisation* stronger than this corpus
    produces by chance? It says nothing about whether the similarity itself is
    unusual -- a corpus where everyone quotes the same press release will fail
    to reject the null while still being full of duplicates.

    The p-value is ``(1 + #{permuted >= observed}) / (permutations + 1)``, so
    it can never be reported as exactly zero.
    """
    config = config or CoordinationConfig()
    records = _fingerprints(corpus, config)
    observed_evidence, _, _ = _accumulate_pairs(records, config)
    observed = sum(
        1
        for entry in observed_evidence.values()
        if len(entry.similarities) >= config.min_edge_weight
    )

    rng = random.Random(config.seed)
    permute = {"shift": _shifted, "shuffle": _shuffled}.get(strategy)
    if permute is None:
        msg = f"unknown null model strategy {strategy!r}; use 'shift' or 'shuffle'"
        raise ValueError(msg)

    counts: list[int] = []
    for _ in range(permutations):
        evidence, _, _ = _accumulate_pairs(permute(records, rng), config)
        counts.append(
            sum(1 for e in evidence.values() if len(e.similarities) >= config.min_edge_weight)
        )

    at_least = sum(1 for count in counts if count >= observed)
    return NullModelResult(
        observed=observed,
        permuted=tuple(counts),
        p_value=(1 + at_least) / (permutations + 1),
        strategy=strategy,
    )


# --------------------------------------------------------------------------
# Account-level features
# --------------------------------------------------------------------------

SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="coord_partner_count",
        description="Number of distinct accounts this account co-published near-duplicates with.",
        rationale=(
            "Operating several accounts from one content queue leaves the same text in "
            "several timelines at once, so partner count grows with the size of the queue "
            "rather than with the size of the audience."
        ),
        limitation=(
            "Anyone amplifying a widely circulated template -- a protest call, a fundraising "
            "appeal, a viral format -- accumulates partners without any coordination at all."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="count",
        value_range=(0.0, None),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="coord_pair_count",
        description="Total near-duplicate co-posting events involving this account.",
        rationale=(
            "Repetition separates a one-off coincidence from a sustained pattern: a shared "
            "template fires once, a shared queue fires repeatedly over the collection window."
        ),
        limitation=(
            "Scales with how much the account posts, so a prolific hobbyist outranks a quiet "
            "automated account; it must be read alongside the account volume features."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="count",
        value_range=(0.0, None),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="coord_max_similarity",
        description="Highest fingerprint similarity reached with any co-publishing partner.",
        rationale=(
            "Near-verbatim reuse is cheap to produce and hard to arrive at independently, so "
            "the ceiling of the similarity distribution is more telling than its mean."
        ),
        limitation=(
            "Quoting the same headline, press release or song lyric produces a perfect score, "
            "and short posts reach high similarity for purely combinatorial reasons."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="coord_median_lag_seconds",
        description="Median delay between this account and its partners on matched posts.",
        rationale=(
            "Humans read, decide and type; dispatch systems do not. Median lags of a few "
            "seconds across many pairs indicate a shared trigger rather than a shared interest."
        ),
        limitation=(
            "Scheduling tools used openly by newsrooms and campaigns produce the same tight "
            "lags, and a push notification can synchronise thousands of genuine readers."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="seconds",
        value_range=(0.0, None),
        higher_is_more_anomalous=False,
    ),
    FeatureSpec(
        name="coord_duplicate_post_ratio",
        description="Share of the account's eligible posts that matched someone else's.",
        rationale=(
            "Normalises for volume: it separates an account whose entire output is recycled "
            "from one that occasionally shares a template alongside original writing."
        ),
        limitation=(
            "Accounts that exist to relay official statements -- mirrors, bulletins, "
            "aggregators -- legitimately sit near 1.0 and are not covert in any way."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="coord_cluster_size",
        description="Size of the Louvain community this account belongs to, else 1.",
        rationale=(
            "Membership in a large, densely connected community is a property of the group "
            "rather than of the individual, which is the level this library reports at."
        ),
        limitation=(
            "Louvain is resolution-dependent and non-deterministic without a fixed seed; the "
            "same graph yields different community sizes under different resolutions."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="count",
        value_range=(1.0, None),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="coord_cluster_density",
        description="Edge density of the account's community, 0 when it has none.",
        rationale=(
            "A near-complete subgraph means every member matched every other member, which "
            "is far harder to reach by shared interest than a sparse chain of overlaps."
        ),
        limitation=(
            "Density falls mechanically as clusters grow, so it cannot be compared across "
            "clusters of very different sizes without normalisation."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
)


class CoordinationExtractor:
    """Account-level view of the co-posting graph, as a feature frame.

    Missing values are meaningful here. An account with no eligible post --
    everything it wrote was too short, or a repost -- gets ``NaN``, because the
    analysis has nothing to say about it. An account that was analysed and
    matched nobody gets ``0``. Collapsing the two would let "we could not look"
    masquerade as "we looked and found nothing".
    """

    name = "coordination"
    specs = SPECS

    def __init__(self, config: CoordinationConfig | None = None) -> None:
        self.config = config or CoordinationConfig()

    def extract(self, corpus: Corpus) -> pd.DataFrame:
        """Compute the coordination features for every account in ``corpus``."""
        frame = empty_frame(corpus, self.specs)
        analysis = _analyse(corpus, self.config)
        clusters = _clusters_from_graph(analysis.graph, self.config)
        result = CoordinationResult(
            graph=analysis.graph,
            clusters=clusters,
            config=self.config,
            stats=analysis.stats,
        )

        eligible_counts: Counter[AccountId] = Counter(r.account_id for r in analysis.records)
        matched_counts: Counter[AccountId] = Counter(
            r.account_id for r in analysis.records if r.post_id in analysis.matched
        )

        for account_id, n_eligible in eligible_counts.items():
            if account_id not in frame.index:  # orphan author, no Account record
                continue
            neighbours = list(result.graph.neighbors(account_id))
            weights = [float(result.graph[account_id][other]["weight"]) for other in neighbours]
            similarities = [
                float(result.graph[account_id][other]["mean_similarity"]) for other in neighbours
            ]
            lags = [
                float(result.graph[account_id][other]["median_lag_seconds"]) for other in neighbours
            ]
            cluster = result.cluster_of(account_id)

            frame.loc[account_id, "coord_partner_count"] = float(len(neighbours))
            frame.loc[account_id, "coord_pair_count"] = float(sum(weights))
            frame.loc[account_id, "coord_max_similarity"] = max(similarities, default=0.0)
            frame.loc[account_id, "coord_median_lag_seconds"] = (
                median(lags) if lags else float("nan")
            )
            frame.loc[account_id, "coord_duplicate_post_ratio"] = (
                matched_counts[account_id] / n_eligible
            )
            frame.loc[account_id, "coord_cluster_size"] = float(cluster.size if cluster else 1)
            frame.loc[account_id, "coord_cluster_density"] = float(
                cluster.density if cluster else 0.0
            )
        return frame


def summarise_clusters(clusters: Iterable[Cluster]) -> dict[str, object]:
    """Corpus-level summary of a set of clusters, for the report header."""
    items = list(clusters)
    sizes = [cluster.size for cluster in items]
    return {
        "n_clusters": len(items),
        "n_accounts_in_clusters": sum(sizes),
        "largest_cluster": max(sizes, default=0),
        "median_cluster_size": median(sizes) if sizes else 0,
    }


def window_sensitivity(
    corpus: Corpus,
    windows: Sequence[timedelta],
    config: CoordinationConfig | None = None,
) -> list[dict[str, object]]:
    """Re-run the analysis across several windows and report how much moves.

    The window is the single most consequential parameter in this module, and
    a result that only exists at one setting of it is not a result. Publishing
    this curve alongside a cluster count is the honest minimum.
    """
    base = config or CoordinationConfig()
    rows: list[dict[str, object]] = []
    for window in windows:
        variant = CoordinationConfig(
            window=window,
            hamming_threshold=base.hamming_threshold,
            shingle_size=base.shingle_size,
            min_chars=base.min_chars,
            include_reposts=base.include_reposts,
            exclude_same_thread=base.exclude_same_thread,
            min_edge_weight=base.min_edge_weight,
            min_cluster_size=base.min_cluster_size,
            resolution=base.resolution,
            seed=base.seed,
            max_candidate_pairs=base.max_candidate_pairs,
        )
        result = detect_coordination(corpus, variant)
        rows.append(
            {
                "window_seconds": window.total_seconds(),
                "n_edges": result.graph.number_of_edges(),
                "n_clusters": len(result.clusters),
                "n_accounts_in_clusters": len(result.clustered_account_ids),
            }
        )
    return rows
