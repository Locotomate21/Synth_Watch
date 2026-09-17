"""Adapters that normalise external sources into the internal schema.

Importing this package registers the adapters that ship with the library, so
``REGISTRY.detect(path)`` works without the caller knowing which module a
given format lives in.
"""

from __future__ import annotations

from synthwatch.ingest.base import REGISTRY, Adapter, AdapterRegistry, IngestReport, LoadResult
from synthwatch.ingest.native import NativeAdapter, parse_timestamp, write_corpus

__all__ = [
    "REGISTRY",
    "Adapter",
    "AdapterRegistry",
    "IngestReport",
    "LoadResult",
    "NativeAdapter",
    "parse_timestamp",
    "write_corpus",
]
