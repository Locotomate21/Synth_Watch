"""Assembling a report: what was analysed, how, what came out, and what it is not.

A :class:`Report` is deliberately self-describing. It carries the corpus
provenance, the ingest drop rate, every configuration that shaped the result,
and each feature's declared limitation alongside its numbers. A figure from
this library that travels without those is not reproducible, and the easiest
way to stop that happening is to make it impossible to export one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from synthwatch import __version__
from synthwatch.detect.account import AccountConfig, AccountExtractor
from synthwatch.detect.base import FeatureExtractor, FeatureRegistry, FeatureSpec
from synthwatch.detect.coordination import (
    CoordinationConfig,
    CoordinationExtractor,
    CoordinationResult,
    detect_coordination,
)
from synthwatch.detect.temporal import TemporalConfig, TemporalExtractor, profile_accounts
from synthwatch.ingest.base import IngestReport
from synthwatch.models import Corpus
from synthwatch.report.privacy import Pseudonymiser
from synthwatch.report.summary import ClusterCard, FeatureSummary, build_cards, summarise_features

__all__ = ["Report", "build_report", "to_json"]

CORPUS_CAVEATS: tuple[str, ...] = (
    "This report describes behaviour observed in one corpus over one window. It "
    "assigns no verdict to any account, and none of its numbers should be read as "
    "one.",
    "Coordination means synchronised near-duplicate publishing. Campaign "
    "volunteers, fandoms, newsrooms and scheduling tools all produce it.",
    "Feature distributions are descriptive. Without labelled data from the same "
    "platform and period, they do not support a prevalence estimate.",
    "Accounts are listed under pseudonyms. This is a speed bump against casual "
    "misuse, not anonymisation: a cluster described here is re-identifiable by "
    "anyone with access to the platform.",
)


@dataclass(frozen=True, slots=True)
class Report:
    """Everything an analysis produced, plus the conditions that produced it."""

    corpus: dict[str, Any]
    configs: dict[str, Any]
    coordination: dict[str, Any] | None
    cards: tuple[ClusterCard, ...]
    features: tuple[FeatureSummary, ...]
    ingest: dict[str, Any] | None = None
    caveats: tuple[str, ...] = CORPUS_CAVEATS
    library_version: str = __version__
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    title: str = "SynthWatch analysis"

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view of the whole report."""
        return {
            "title": self.title,
            "library_version": self.library_version,
            "generated_at": self.generated_at.isoformat(),
            "corpus": self.corpus,
            "ingest": self.ingest,
            "configs": self.configs,
            "coordination": self.coordination,
            "clusters": [card.as_dict() for card in self.cards],
            "features": [summary.as_dict() for summary in self.features],
            "caveats": list(self.caveats),
        }


def build_report(
    corpus: Corpus,
    *,
    coordination_config: CoordinationConfig | None = None,
    temporal_config: TemporalConfig | None = None,
    account_config: AccountConfig | None = None,
    ingest_report: IngestReport | None = None,
    null_model_permutations: int = 0,
    pseudonymise: bool = True,
    pseudonym_salt: str | None = None,
    include_examples: bool = True,
    title: str = "SynthWatch analysis",
) -> tuple[Report, pd.DataFrame]:
    """Run the pipeline over ``corpus`` and assemble a report.

    Args:
        corpus: The corpus to analyse.
        coordination_config: Co-posting parameters; defaults are documented on
            :class:`~synthwatch.detect.coordination.CoordinationConfig`.
        temporal_config: Posting-rhythm parameters.
        account_config: Profile-feature parameters.
        ingest_report: The load report, so the drop rate travels with the
            findings.
        null_model_permutations: Permutations for the coordination significance
            check. Zero skips it, and the report says so.
        pseudonymise: Whether exported account ids are replaced by pseudonyms.
        pseudonym_salt: Fixed salt, for reports that need to be comparable with
            each other. Omit for a fresh random one.
        include_examples: Whether cluster cards cite the post ids behind them.
        title: Title carried into the exports.

    Returns:
        The report, and the account feature matrix it was summarised from. The
        matrix is returned rather than embedded: it holds per-account values,
        which is exactly what a report must not publish.
    """
    coordination_config = coordination_config or CoordinationConfig()
    temporal_config = temporal_config or TemporalConfig()
    account_config = account_config or AccountConfig()

    extractors: list[FeatureExtractor] = [
        AccountExtractor(account_config),
        TemporalExtractor(temporal_config),
    ]
    frames = [extractor.extract(corpus) for extractor in extractors]
    features = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    result: CoordinationResult = detect_coordination(
        corpus,
        coordination_config,
        null_model_permutations=null_model_permutations,
    )
    coordination_extractor = CoordinationExtractor(coordination_config)
    coordination_frame = coordination_extractor.extract(corpus)
    if not coordination_frame.empty:
        features = pd.concat([features, coordination_frame], axis=1)

    pseudonymiser = (
        Pseudonymiser(salt=pseudonym_salt, enabled=pseudonymise)
        if pseudonym_salt
        else Pseudonymiser(enabled=pseudonymise)
    )
    cards = build_cards(
        result.clusters,
        corpus=corpus,
        features=features,
        profiles=profile_accounts(corpus, temporal_config),
        pseudonymiser=pseudonymiser,
        include_examples=include_examples,
    )

    registry = FeatureRegistry.from_extractors([*extractors, coordination_extractor])
    specs: list[FeatureSpec] = list(registry.specs)
    caveats = list(CORPUS_CAVEATS)
    if not pseudonymise:
        caveats[-1] = (
            "Accounts are listed under their real identifiers because "
            "pseudonymisation was disabled for this report."
        )
    if null_model_permutations == 0:
        caveats.append(
            "No null model was run, so the cluster count has nothing to be compared "
            "against. Re-run with permutations before citing it."
        )

    report = Report(
        corpus=corpus.summary(),
        ingest=ingest_report.as_dict() if ingest_report else None,
        configs={
            "coordination": coordination_config.as_dict(),
            "temporal": temporal_config.as_dict(),
            "account": account_config.as_dict(),
            "null_model_permutations": null_model_permutations,
            "pseudonymised": pseudonymise,
        },
        coordination=_coordination_section(result.as_dict()),
        cards=tuple(cards),
        features=tuple(summarise_features(features, specs)),
        caveats=tuple(caveats),
        title=title,
    )
    return report, features


def _coordination_section(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip cluster identities out of the embedded coordination result.

    The raw result lists each cluster's real account ids. Cluster membership is
    described exactly once in a report -- by the cards, pseudonymised -- so the
    copy here would be a silent way around the report layer's only real
    promise.
    """
    return {key: value for key, value in payload.items() if key != "clusters"}


def to_json(report: Report, path: Path | None = None, *, indent: int = 2) -> str:
    """Serialise a report, writing it to ``path`` when one is given."""
    payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=indent, default=str)
    if path is not None:
        path.write_text(payload, encoding="utf-8")
    return payload


def feature_names(specs: Sequence[FeatureSpec]) -> list[str]:
    """Declared feature names, in declaration order."""
    return [spec.name for spec in specs]
