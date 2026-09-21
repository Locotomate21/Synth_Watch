"""Profile an unknown file before trying to analyse it.

The first thing anyone does with a downloaded dataset is discover that its
columns are not what the documentation said. This module answers that question
directly: what is in the file, which columns the schema recognises, which ones
would be dropped into ``extra``, and what would happen to a sample of rows if
they were loaded for real.

Nothing here parses the whole file. A published archive can be a hundred
gigabytes, and the shape of the first few thousand rows is what a person needs
before deciding anything.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from synthwatch.ingest.native import (
    DEFAULT_ACCOUNT_ALIASES,
    DEFAULT_POST_ALIASES,
    NativeAdapter,
)
from synthwatch.models import Account, Post

__all__ = ["ColumnProfile", "FileProfile", "inspect_file"]

SAMPLE_VALUES = 3
"""Distinct example values kept per column, enough to recognise a format."""

DEFAULT_SAMPLE_ROWS = 2000
"""Rows read for a profile. Large enough to be representative, small enough
that pointing this at a 100 GB archive returns immediately."""


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """What one column of a source file looks like.

    Attributes:
        name: The column as written in the file.
        maps_to: Schema field it would populate, or ``None`` if unrecognised.
        n_present: Rows in the sample where the column had a value.
        n_empty: Rows where it was blank.
        examples: A few distinct values, for recognising the format by eye.
    """

    name: str
    maps_to: str | None
    n_present: int
    n_empty: int
    examples: tuple[str, ...]

    @property
    def fill_rate(self) -> float:
        """Share of sampled rows where the column carried something."""
        total = self.n_present + self.n_empty
        return self.n_present / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        """Serialisable view."""
        return {
            "name": self.name,
            "maps_to": self.maps_to,
            "fill_rate": round(self.fill_rate, 3),
            "examples": list(self.examples),
        }


@dataclass(frozen=True, slots=True)
class FileProfile:
    """The shape of a source file, and what loading it would do.

    Attributes:
        path: The file profiled.
        n_sampled: Rows read.
        columns: One profile per column, in file order.
        recognised: Columns that map onto a schema field.
        unrecognised: Columns that would be preserved in ``extra``.
        missing_required: Schema fields with no column feeding them.
        trial_skips: What a trial load of the sample skipped, and why.
        n_trial_loaded: Records the trial load produced.
    """

    path: Path
    n_sampled: int
    columns: tuple[ColumnProfile, ...]
    recognised: tuple[str, ...]
    unrecognised: tuple[str, ...]
    missing_required: tuple[str, ...]
    trial_skips: Mapping[str, int]
    n_trial_loaded: int

    @property
    def usable(self) -> bool:
        """Whether a load would produce anything at all."""
        return not self.missing_required and self.n_trial_loaded > 0

    def as_dict(self) -> dict[str, object]:
        """Serialisable view."""
        return {
            "path": str(self.path),
            "n_sampled": self.n_sampled,
            "usable": self.usable,
            "recognised": list(self.recognised),
            "unrecognised": list(self.unrecognised),
            "missing_required": list(self.missing_required),
            "n_trial_loaded": self.n_trial_loaded,
            "trial_skips": dict(self.trial_skips),
            "columns": [column.as_dict() for column in self.columns],
        }


REQUIRED_POST_FIELDS = ("post_id", "account_id", "created_at")
"""Without these a row cannot become a :class:`~synthwatch.models.Post`."""


def _sample_values(values: Sequence[str]) -> tuple[str, ...]:
    """A few distinct, non-empty values, truncated for display."""
    seen: list[str] = []
    for value in values:
        text = value.strip()
        if text and text not in seen:
            seen.append(text[:80])
        if len(seen) == SAMPLE_VALUES:
            break
    return tuple(seen)


def inspect_file(
    path: Path,
    *,
    kind: str = "posts",
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    adapter: NativeAdapter | None = None,
) -> FileProfile:
    """Profile ``path`` without committing to loading all of it.

    Args:
        path: File to inspect.
        kind: ``"posts"`` or ``"accounts"``, which alias table to check against.
        sample_rows: How many rows to read.
        adapter: Adapter whose aliases and declarations are used for the trial
            load, so the profile reflects the options a real load would use.

    Returns:
        The profile, including what a trial load of the sample would skip.
    """
    adapter = adapter or NativeAdapter()
    rows = list(adapter.read_records(path))[:sample_rows]
    aliases = DEFAULT_POST_ALIASES if kind == "posts" else DEFAULT_ACCOUNT_ALIASES
    extra_aliases = adapter.post_aliases if kind == "posts" else adapter.account_aliases
    known = set(Post.model_fields if kind == "posts" else Account.model_fields) - {"extra"}

    names: list[str] = []
    for row in rows:
        names.extend(name for name in row if name not in names)

    columns: list[ColumnProfile] = []
    for name in names:
        values = [str(row.get(name) or "") for row in rows]
        canonical = name.strip()
        target = (
            canonical
            if canonical in known
            else extra_aliases.get(canonical.casefold(), aliases.get(canonical.casefold()))
        )
        columns.append(
            ColumnProfile(
                name=name,
                maps_to=target if target in known else None,
                n_present=sum(1 for value in values if value.strip()),
                n_empty=sum(1 for value in values if not value.strip()),
                examples=_sample_values(values),
            )
        )

    mapped = {column.maps_to for column in columns if column.maps_to}
    required = REQUIRED_POST_FIELDS if kind == "posts" else ("account_id",)
    trial = adapter.from_records(
        path.name,
        rows if kind == "posts" else (),
        () if kind == "posts" else rows,
    )
    loaded = trial.report.n_posts if kind == "posts" else trial.report.n_accounts

    return FileProfile(
        path=path,
        n_sampled=len(rows),
        columns=tuple(columns),
        recognised=tuple(sorted(mapped)),
        unrecognised=tuple(c.name for c in columns if c.maps_to is None),
        missing_required=tuple(name for name in required if name not in mapped),
        trial_skips=dict(Counter(trial.report.skip_reasons)),
        n_trial_loaded=loaded,
    )


def render_profile(profile: FileProfile) -> str:
    """Render a profile as plain text for a terminal."""
    lines = [
        f"{profile.path.name}: sampled {profile.n_sampled} rows",
        "",
        f"{'column':<28} {'maps to':<22} {'fill':>5}  example",
    ]
    for column in profile.columns:
        target = column.maps_to or "-- extra --"
        example = column.examples[0] if column.examples else ""
        lines.append(
            f"{column.name[:28]:<28} {target:<22} {column.fill_rate:>4.0%}  {example[:40]}"
        )

    lines.append("")
    if profile.missing_required:
        lines.append(f"MISSING required field(s): {', '.join(profile.missing_required)}")
        lines.append("  map them with --alias source_column=schema_field")
    lines.append(f"trial load of the sample: {profile.n_trial_loaded} record(s)")
    for reason, count in sorted(profile.trial_skips.items()):
        lines.append(f"  skipped {count}: {reason}")
    if not profile.trial_skips and not profile.missing_required:
        lines.append("  nothing skipped")
    return "\n".join(lines)
