"""Aggregate metrics, cluster cards and export formats.

This is the layer where the library's scope limits are enforced rather than
merely documented: reports describe corpora and clusters, account identifiers
are pseudonymised on the way out, and every figure travels with the parameters
and the known limitation that qualify it.
"""

from __future__ import annotations

from synthwatch.report.html import to_html
from synthwatch.report.privacy import Pseudonymiser
from synthwatch.report.report import Report, build_report, to_json
from synthwatch.report.summary import ClusterCard, FeatureSummary, build_cards, summarise_features

__all__ = [
    "ClusterCard",
    "FeatureSummary",
    "Pseudonymiser",
    "Report",
    "build_cards",
    "build_report",
    "summarise_features",
    "to_html",
    "to_json",
]
