"""Regenerate ``docs/features.md`` from the declared feature specs."""

from __future__ import annotations

from pathlib import Path

from synthwatch.detect.base import FeatureRegistry, render_catalogue
from synthwatch.detect.coordination import CoordinationExtractor

EXTRACTORS = [CoordinationExtractor()]
"""Every extractor in the pipeline. Add new ones here as they land."""

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "features.md"


def main() -> None:
    """Write the catalogue to ``docs/features.md``."""
    catalogue = render_catalogue(FeatureRegistry.from_extractors(EXTRACTORS))
    OUTPUT.write_text(catalogue, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
