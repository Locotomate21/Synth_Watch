"""Aggregate views: feature distributions and cluster cards.

Nothing in this module emits a per-account number. That is the whole design:
the analysis produces an account-level feature matrix, and everything that
leaves the library is either a distribution over the corpus or a description of
a cluster. An account's own row stays in the analyst's session.

The distributions are reported with their missingness, because a feature
computed for 12% of a corpus and a feature computed for all of it support very
different claims, and a median alone hides the difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import pandas as pd

from synthwatch.detect.base import FeatureSpec
from synthwatch.detect.coordination import Cluster
from synthwatch.detect.temporal import TemporalProfile
from synthwatch.models import Corpus
from synthwatch.report.privacy import Pseudonymiser
from synthwatch.types import AccountId

__all__ = ["ClusterCard", "FeatureSummary", "build_cards", "summarise_features"]

PERCENTILES = (0.1, 0.5, 0.9)
"""Reported quantiles. Deliberately not the extremes: a maximum is one account."""


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    """The distribution of one feature across a corpus.

    Attributes:
        spec: The feature's declaration, so a report carries its rationale and
            its known limitation next to its numbers.
        n_measured: Accounts the feature could be computed for.
        n_missing: Accounts whose metadata did not support it.
        p10: Tenth percentile of the measured values.
        median: Median of the measured values.
        p90: Ninetieth percentile of the measured values.
    """

    spec: FeatureSpec
    n_measured: int
    n_missing: int
    p10: float | None
    median: float | None
    p90: float | None

    @property
    def coverage(self) -> float:
        """Share of accounts the feature was measurable for."""
        total = self.n_measured + self.n_missing
        return self.n_measured / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, rationale and limitation included."""
        return {
            "name": self.spec.name,
            "unit": self.spec.unit,
            "description": self.spec.description,
            "rationale": self.spec.rationale,
            "limitation": self.spec.limitation,
            "direction": self.spec.higher_is_more_anomalous,
            "n_measured": self.n_measured,
            "n_missing": self.n_missing,
            "coverage": round(self.coverage, 4),
            "p10": _round(self.p10),
            "median": _round(self.median),
            "p90": _round(self.p90),
        }


def _round(value: float | None) -> float | None:
    """Round for display, preserving the difference between 0 and unknown."""
    return None if value is None else round(value, 4)


def summarise_features(frame: pd.DataFrame, specs: Sequence[FeatureSpec]) -> list[FeatureSummary]:
    """Summarise every declared feature present in ``frame``.

    Features declared but absent from the frame are skipped rather than
    reported as empty: their extractor did not run, which is a different
    statement from "no account could be measured".
    """
    summaries: list[FeatureSummary] = []
    for spec in specs:
        if spec.name not in frame.columns:
            continue
        column = frame[spec.name]
        measured = column.dropna()
        quantiles = (
            [float(measured.quantile(q)) for q in PERCENTILES] if len(measured) else [None] * 3
        )
        summaries.append(
            FeatureSummary(
                spec=spec,
                n_measured=len(measured),
                n_missing=int(column.isna().sum()),
                p10=quantiles[0],
                median=quantiles[1],
                p90=quantiles[2],
            )
        )
    return summaries


@dataclass(frozen=True, slots=True)
class ClusterCard:
    """What a report says about one coordinated cluster.

    A card describes a group. It carries no per-account score, and its member
    list is pseudonymous unless the analyst turned that off. The caveat is a
    field rather than a footnote so that it cannot be dropped by whoever
    reformats the output.
    """

    cluster_id: str
    size: int
    members: tuple[str, ...]
    density: float
    mean_similarity: float
    median_lag_seconds: float
    total_pairs: int
    example_post_pairs: tuple[tuple[str, str], ...]
    median_account_age_days: float | None
    median_circadian_entropy: float | None
    quiet_window_free_members: int | None
    caveat: str

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for JSON and HTML export."""
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "members": list(self.members),
            "density": round(self.density, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "median_lag_seconds": round(self.median_lag_seconds, 1),
            "total_pairs": self.total_pairs,
            "example_post_pairs": [list(pair) for pair in self.example_post_pairs],
            "median_account_age_days": _round(self.median_account_age_days),
            "median_circadian_entropy": _round(self.median_circadian_entropy),
            "quiet_window_free_members": self.quiet_window_free_members,
            "caveat": self.caveat,
        }


CLUSTER_CAVEAT = (
    "These accounts published near-identical content within the analysis window, "
    "repeatedly. That is a behaviour, not an intent and not an identity. Shared "
    "message guides, fandoms, syndicated wire copy and ordinary scheduling tools "
    "produce the same pattern. Treat this as a question to investigate."
)

QUIET_WINDOW_THRESHOLD = 0.15
"""Above this share of posts in its quietest six hours, an account has no night."""


def build_cards(
    clusters: Sequence[Cluster],
    *,
    corpus: Corpus,
    features: pd.DataFrame | None = None,
    profiles: dict[AccountId, TemporalProfile] | None = None,
    pseudonymiser: Pseudonymiser | None = None,
    include_examples: bool = True,
) -> list[ClusterCard]:
    """Turn coordination clusters into report cards.

    Args:
        clusters: Clusters from ``detect.coordination``.
        corpus: The analysed corpus, for the record counts behind each card.
        features: Optional account feature frame, used only for medians across
            the cluster -- never for a per-member value.
        profiles: Optional temporal profiles, same restriction.
        pseudonymiser: Applied to every member id. Members are listed under
            their real ids only if this is explicitly disabled.
        include_examples: Whether to cite the post ids behind the cluster.
    """
    # `pseudonymiser or Pseudonymiser()` would be wrong: a fresh one has no
    # entries yet, __len__ makes it falsy, and an explicit salt or a disabled
    # mapping would be silently thrown away.
    mask = pseudonymiser if pseudonymiser is not None else Pseudonymiser()
    cards: list[ClusterCard] = []
    for cluster in clusters:
        members = [
            account_id
            for account_id in cluster.account_ids
            if account_id in corpus.posts_by_account
        ]
        cards.append(
            ClusterCard(
                cluster_id=cluster.cluster_id,
                size=cluster.size,
                members=tuple(mask(account_id) for account_id in cluster.account_ids),
                density=cluster.density,
                mean_similarity=cluster.mean_similarity,
                median_lag_seconds=cluster.median_lag_seconds,
                total_pairs=cluster.total_pairs,
                example_post_pairs=cluster.examples if include_examples else (),
                median_account_age_days=_cluster_median(
                    features, cluster.account_ids, "acct_age_days"
                ),
                median_circadian_entropy=_cluster_median(
                    features, cluster.account_ids, "temp_circadian_entropy"
                ),
                quiet_window_free_members=_quiet_window_free(profiles, members),
                caveat=CLUSTER_CAVEAT,
            )
        )
    return cards


def _cluster_median(
    features: pd.DataFrame | None, account_ids: Sequence[AccountId], column: str
) -> float | None:
    """Median of one feature across a cluster, or ``None`` if unmeasured."""
    if features is None or column not in features.columns:
        return None
    present = [a for a in account_ids if a in features.index]
    if not present:
        return None
    values = features.loc[present, column].dropna()
    return float(values.median()) if len(values) else None


def _quiet_window_free(
    profiles: dict[AccountId, TemporalProfile] | None, account_ids: Sequence[AccountId]
) -> int | None:
    """How many members post through their own quietest hours."""
    if profiles is None:
        return None
    shares = [
        profiles[account_id].quiet_share for account_id in account_ids if account_id in profiles
    ]
    if not shares:
        return None
    return sum(share > QUIET_WINDOW_THRESHOLD for share in shares)


def median_or_none(values: Sequence[float]) -> float | None:
    """Median of a possibly empty sequence."""
    return median(values) if values else None
