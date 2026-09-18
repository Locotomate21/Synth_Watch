"""Small numeric primitives shared by more than one extractor.

Kept separate from ``detect.base`` so that the contracts module stays about
contracts, and separate from any one extractor so that importing a helper does
not drag an unrelated feature family along with it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = ["safe_ratio", "shannon_entropy"]


def shannon_entropy(counts: Sequence[int | float], *, bins: int | None = None) -> float:
    """Normalised Shannon entropy of a histogram, in ``[0, 1]``.

    Args:
        counts: Bin counts. Empty bins are fine and carry no information.
        bins: Denominator for normalisation, defaulting to ``len(counts)``.
            Pass it explicitly when the histogram is sparse but the number of
            possible bins is known, so that two subjects stay comparable.

    Returns:
        ``0.0`` when everything sits in one bin, ``1.0`` for a perfectly flat
        distribution, and ``0.0`` when there is nothing to measure.
    """
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    width = bins if bins is not None else len(counts)
    if width <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts if c > 0)
    # A single populated bin yields -0.0, and `max(-0.0, 0.0)` keeps the sign.
    # abs() is safe because entropy is non-negative by construction.
    return abs(entropy) / math.log2(width)


def character_entropy(text: str) -> float:
    """Normalised entropy of the characters in ``text``.

    Normalised against the number of *distinct characters present* rather than
    an alphabet size, so that short strings are not penalised for being short.
    """
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1
    return shannon_entropy(list(counts.values()))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Divide, returning ``None`` rather than a number that means nothing.

    ``None`` propagates when either side is missing, and a zero denominator
    yields ``None`` instead of infinity or an arbitrary sentinel. Downstream
    that becomes ``NaN``, which the ensemble reads as "unknown" rather than as
    a measured extreme.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def mean(values: Iterable[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty input."""
    items = list(values)
    return sum(items) / len(items) if items else None
