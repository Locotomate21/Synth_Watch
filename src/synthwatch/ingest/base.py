"""Adapter contract for turning a data source into a :class:`~synthwatch.models.Corpus`.

An adapter is the only place in the library allowed to know about a platform's
vocabulary. It answers three questions and nothing else: which record is an
account, which is a post, and what the timestamps actually mean. Anything it
cannot map goes into ``extra`` rather than being dropped, so a field that turns
out to matter later can be recovered without re-collecting the data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from synthwatch.models import Corpus
from synthwatch.types import Platform


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What happened during a load, kept alongside the data.

    Dropped and repaired records are counted rather than logged away, because
    the drop rate is itself a finding: an adapter that silently discards 30% of
    a collection will produce confident and meaningless coordination scores.
    """

    source: str
    n_accounts: int = 0
    n_posts: int = 0
    n_skipped: int = 0
    skip_reasons: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view for the report layer."""
        return {
            "source": self.source,
            "n_accounts": self.n_accounts,
            "n_posts": self.n_posts,
            "n_skipped": self.n_skipped,
            "skip_reasons": dict(self.skip_reasons),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class LoadResult:
    """A corpus plus the provenance of the load that produced it."""

    corpus: Corpus
    report: IngestReport


@runtime_checkable
class Adapter(Protocol):
    """Reads one source format into the internal schema."""

    name: str
    platform: Platform

    def sniff(self, path: Path) -> bool:
        """Whether this adapter recognises ``path`` (cheap check, no full read)."""
        ...

    def load(self, path: Path) -> LoadResult:
        """Read ``path`` into a corpus.

        Per-source options -- column aliases, an assumed timezone, the platform
        to stamp on records -- are constructor arguments of the concrete
        adapter, not arguments here. That keeps the options that shaped a load
        attached to the object that performed it, so a report can record them.

        Implementations must not perform network calls: fetching is a separate
        concern from parsing, so that every analysis can be re-run offline from
        the archived file.
        """
        ...


class AdapterRegistry:
    """Name-to-adapter lookup used by the CLI and the ingest entry points."""

    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}

    def register(self, adapter: Adapter) -> None:
        """Add ``adapter``, refusing to shadow an existing name."""
        if adapter.name in self._adapters:
            msg = f"adapter {adapter.name!r} is already registered"
            raise ValueError(msg)
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> Adapter:
        """Look up an adapter by name."""
        try:
            return self._adapters[name]
        except KeyError:
            known = ", ".join(sorted(self._adapters)) or "<none>"
            msg = f"unknown adapter {name!r}; registered: {known}"
            raise KeyError(msg) from None

    def detect(self, path: Path) -> Adapter | None:
        """First registered adapter that recognises ``path``, if any."""
        return next((a for a in self._adapters.values() if a.sniff(path)), None)

    def __iter__(self) -> Iterator[Adapter]:
        """Iterate over registered adapters in registration order."""
        return iter(self._adapters.values())

    @property
    def names(self) -> Sequence[str]:
        """Registered adapter names."""
        return tuple(self._adapters)


REGISTRY = AdapterRegistry()
"""Process-wide registry. Adapters register themselves on import."""
