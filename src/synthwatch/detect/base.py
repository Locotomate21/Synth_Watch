"""Contracts every feature extractor obeys.

The central object here is :class:`FeatureSpec`. A feature is not considered to
exist until it is declared with a *rationale* (why this signal should correlate
with automation or synthetic text) and a *limitation* (the population it is
known to misclassify). ``tests/detect/test_feature_spec.py`` fails the build if
either is missing, and ``docs/features.md`` is generated from these specs, so
the documentation cannot drift away from the code.

The rule exists because the failure mode of this kind of tool is not a bad
F1 score, it is a plausible-looking number whose caveats were never written
down and therefore never reached the person reading the report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from synthwatch.models import Corpus
from synthwatch.types import FeatureLevel

MIN_DOC_CHARS = 40
"""Minimum length for a rationale or limitation. A deterrent against "n/a"."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declaration of a single feature, including what it cannot do.

    Attributes:
        name: Column name in the feature matrix. Snake case, prefixed by
            extractor family (``text_``, ``acct_``, ``temp_``, ``coord_``).
        description: What is computed, in one sentence.
        rationale: Why this should carry signal. Cite the literature where
            there is any.
        limitation: The known failure mode -- who gets a high score while
            being perfectly ordinary. Written for the report's reader, not
            for the developer.
        level: Unit of analysis the value describes.
        unit: Unit of the value (``ratio``, ``days``, ``bits``, ``count``...).
        value_range: Theoretical bounds, ``None`` where unbounded.
        higher_is_more_anomalous: Direction of the signal, or ``None`` when the
            relationship is non-monotonic and only the model should read it.
        references: Bibliographic pointers.
    """

    name: str
    description: str
    rationale: str
    limitation: str
    level: FeatureLevel
    unit: str = "ratio"
    value_range: tuple[float | None, float | None] = (None, None)
    higher_is_more_anomalous: bool | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the documentation contract at construction time."""
        if not self.name.islower() or " " in self.name:
            msg = f"feature name must be lower snake_case: {self.name!r}"
            raise ValueError(msg)
        for attribute in ("rationale", "limitation"):
            text = getattr(self, attribute)
            if len(text.strip()) < MIN_DOC_CHARS:
                msg = (
                    f"feature {self.name!r}: {attribute} must be a real sentence "
                    f"of at least {MIN_DOC_CHARS} characters"
                )
                raise ValueError(msg)


@runtime_checkable
class FeatureExtractor(Protocol):
    """A family of features computed over a corpus.

    Implementations must be pure with respect to the corpus: calling
    :meth:`extract` twice on the same input returns the same frame, and no
    state leaks between calls. This is what makes the pipeline reproducible
    from a corpus file plus a config hash.
    """

    name: str
    specs: Sequence[FeatureSpec]

    def extract(self, corpus: Corpus) -> pd.DataFrame:
        """Compute the features for every account in ``corpus``.

        Returns:
            A frame indexed by ``account_id``, with one column per spec.
            Missing values are ``NaN`` rather than imputed: whether an absent
            signal means "normal" or "unknown" is a modelling decision that
            belongs to the ensemble, not to the extractor.
        """
        ...


@dataclass(frozen=True, slots=True)
class FeatureRegistry:
    """The set of specs declared by the extractors in a pipeline."""

    specs: tuple[FeatureSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_extractors(cls, extractors: Sequence[FeatureExtractor]) -> FeatureRegistry:
        """Collect specs from ``extractors``, rejecting duplicate names."""
        seen: dict[str, str] = {}
        collected: list[FeatureSpec] = []
        for extractor in extractors:
            for spec in extractor.specs:
                if spec.name in seen:
                    msg = (
                        f"duplicate feature {spec.name!r} declared by "
                        f"{extractor.name!r} and {seen[spec.name]!r}"
                    )
                    raise ValueError(msg)
                seen[spec.name] = extractor.name
                collected.append(spec)
        return cls(tuple(collected))

    @property
    def names(self) -> tuple[str, ...]:
        """Declared feature names, in declaration order."""
        return tuple(spec.name for spec in self.specs)

    def to_records(self) -> list[dict[str, object]]:
        """Serialisable view, used by the report layer and the docs generator."""
        return [
            {
                "name": s.name,
                "level": s.level.value,
                "unit": s.unit,
                "description": s.description,
                "rationale": s.rationale,
                "limitation": s.limitation,
                "direction": s.higher_is_more_anomalous,
                "references": list(s.references),
            }
            for s in self.specs
        ]


def empty_frame(corpus: Corpus, specs: Sequence[FeatureSpec]) -> pd.DataFrame:
    """Build the NaN-filled frame an extractor fills in.

    Guarantees that every account in the corpus gets a row, including silent
    ones, so that feature frames from different extractors align on the index.
    """
    index = pd.Index(
        sorted({a.account_id for a in corpus.accounts} | {p.account_id for p in corpus.posts}),
        name="account_id",
    )
    return pd.DataFrame(float("nan"), index=index, columns=[s.name for s in specs])


DIRECTIONS = {
    True: "higher = more anomalous",
    False: "lower = more anomalous",
    None: "non-monotonic",
}


def render_catalogue(registry: FeatureRegistry) -> str:
    """Render the feature catalogue as Markdown.

    ``docs/features.md`` is written by ``scripts/gen_feature_docs.py`` from
    this function, and a test compares the file against the registry. A
    feature whose documentation changes in code and not in the docs fails the
    build, which is the only reliable way to keep the two honest.
    """
    lines = [
        "# Feature catalogue",
        "",
        "Generated from the `FeatureSpec` declarations in `synthwatch.detect` by",
        "`scripts/gen_feature_docs.py`. Do not edit by hand.",
        "",
        "Every feature declares why it should carry signal and who it misclassifies",
        "while behaving perfectly normally. The second half is the one that has to",
        "reach the reader of a report.",
        "",
    ]
    for spec in registry.specs:
        arrow = DIRECTIONS[spec.higher_is_more_anomalous]
        lines += [
            f"## `{spec.name}`",
            "",
            spec.description,
            "",
            f"- **Level**: {spec.level.value} · **Unit**: {spec.unit} · **Direction**: {arrow}",
            f"- **Rationale**: {spec.rationale}",
            f"- **Known limitation**: {spec.limitation}",
        ]
        if spec.references:
            lines.append("- **References**: " + "; ".join(spec.references))
        lines.append("")
    return "\n".join(lines)
