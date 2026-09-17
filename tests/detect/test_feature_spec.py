"""The documentation contract is a test, not a convention."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from synthwatch.detect.base import (
    FeatureRegistry,
    FeatureSpec,
    empty_frame,
    render_catalogue,
)
from synthwatch.models import build_corpus
from synthwatch.types import FeatureLevel
from tests.conftest import make_account, make_post

VALID: dict[str, Any] = {
    "name": "coord_max_cosim",
    "description": "Highest content similarity with any other account in the window.",
    "rationale": (
        "Independent authors converge on identical wording far less often than "
        "accounts fed from a shared copy deck or template."
    ),
    "limitation": (
        "Quoting the same press release, meme format or breaking-news headline "
        "produces the same signature as coordination."
    ),
    "level": FeatureLevel.ACCOUNT,
}


def test_spec_accepts_a_documented_feature():
    assert FeatureSpec(**VALID).name == "coord_max_cosim"


@pytest.mark.parametrize("missing", ["rationale", "limitation"])
def test_spec_rejects_an_undocumented_feature(missing: str):
    with pytest.raises(ValueError, match=missing):
        FeatureSpec(**{**VALID, missing: "n/a"})


def test_spec_rejects_a_badly_named_feature():
    with pytest.raises(ValueError, match="snake_case"):
        FeatureSpec(**{**VALID, "name": "Coord Max CoSim"})


def test_registry_rejects_duplicate_feature_names():
    class Fake:
        def __init__(self, name: str) -> None:
            self.name = name
            self.specs: Sequence[FeatureSpec] = (FeatureSpec(**VALID),)

        def extract(self, corpus):  # pragma: no cover - protocol filler
            raise NotImplementedError

    with pytest.raises(ValueError, match="duplicate feature"):
        FeatureRegistry.from_extractors([Fake("text"), Fake("coordination")])


def test_empty_frame_covers_silent_and_orphan_accounts():
    corpus = build_corpus(
        [make_account("a1"), make_account("silent")],
        [make_post("p1", "a1")],
    )
    frame = empty_frame(corpus, [FeatureSpec(**VALID)])
    assert list(frame.index) == ["a1", "silent"]
    assert frame["coord_max_cosim"].isna().all()


def test_feature_docs_are_in_sync_with_the_registry():
    # docs/features.md is generated. If this fails, run:
    #     python scripts/gen_feature_docs.py
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "gen_feature_docs", root / "scripts" / "gen_feature_docs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected = render_catalogue(FeatureRegistry.from_extractors(module.EXTRACTORS))
    actual = (root / "docs" / "features.md").read_text(encoding="utf-8")
    assert actual == expected, "docs/features.md is stale: run scripts/gen_feature_docs.py"
