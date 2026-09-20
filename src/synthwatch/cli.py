"""Command line entry point.

One command does the whole pipeline -- load, analyse, report -- because the
alternative is a README full of Python snippets that drift from the code. The
flags are the decisions that change results, and every one of them is written
into the report that comes out.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

from synthwatch import __version__
from synthwatch.detect.account import AccountExtractor
from synthwatch.detect.base import FeatureRegistry, render_catalogue
from synthwatch.detect.coordination import CoordinationConfig, CoordinationExtractor
from synthwatch.detect.ensemble import (
    AutomationEnsemble,
    EnsembleConfig,
    summarise_probabilities,
)
from synthwatch.detect.temporal import TemporalConfig, TemporalExtractor
from synthwatch.ingest.labelled import attach_labels, read_label_table
from synthwatch.ingest.native import NativeAdapter, timezone_of, write_corpus
from synthwatch.report.html import to_html
from synthwatch.report.report import build_report, to_json
from synthwatch.types import Platform

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="synthwatch",
        description=(
            "Measure automation and coordination in political conversation. "
            "Outputs describe corpora and clusters, never individual accounts."
        ),
    )
    parser.add_argument("--version", action="version", version=f"synthwatch {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyse = subcommands.add_parser(
        "analyse", help="load a corpus, run the analysis, write a report"
    )
    analyse.add_argument("posts", type=Path, help="posts file (.csv, .tsv, .json, .jsonl)")
    analyse.add_argument("--accounts", type=Path, help="account metadata file")
    analyse.add_argument("--html", type=Path, help="write an HTML report here")
    analyse.add_argument("--json", type=Path, help="write a JSON report here")
    analyse.add_argument(
        "--features",
        type=Path,
        help=(
            "write the per-account feature matrix here. This file holds "
            "account-level values and is not a report; it stays with the analyst."
        ),
    )
    analyse.add_argument(
        "--platform",
        choices=[p.value for p in Platform],
        default=Platform.GENERIC.value,
        help="platform to stamp on records that do not declare one",
    )
    analyse.add_argument(
        "--assume-timezone",
        type=float,
        metavar="HOURS",
        help=(
            "UTC offset to attach to naive timestamps. Without it, rows with no "
            "offset are skipped and counted rather than guessed at."
        ),
    )
    analyse.add_argument(
        "--window",
        type=float,
        default=15.0,
        metavar="MINUTES",
        help="co-posting window (default: 15). The single most consequential setting.",
    )
    analyse.add_argument(
        "--min-edge-weight",
        type=int,
        default=2,
        help="co-posts required before two accounts get an edge (default: 2)",
    )
    analyse.add_argument(
        "--permutations",
        type=int,
        default=0,
        help=(
            "null model permutations (default: 0, which skips it). Use at least "
            "50 for anything you intend to cite."
        ),
    )
    analyse.add_argument(
        "--min-posts",
        type=int,
        default=20,
        help="posts required before temporal features are reported (default: 20)",
    )
    analyse.add_argument(
        "--no-pseudonyms",
        action="store_true",
        help="list real account identifiers in the report instead of pseudonyms",
    )
    analyse.add_argument("--salt", help="fixed pseudonym salt, for comparable reports")
    analyse.add_argument(
        "--no-examples",
        action="store_true",
        help=(
            "omit the post ids cited as evidence behind each cluster. They are "
            "real by design -- they are what an analyst verifies against -- so "
            "withhold them when the report travels further than the corpus does."
        ),
    )
    analyse.add_argument("--title", default="SynthWatch analysis", help="report title")
    analyse.add_argument("--strict", action="store_true", help="fail on the first bad row")

    labels = subcommands.add_parser(
        "labels", help="check how far an annotation file reaches into a corpus"
    )
    labels.add_argument("posts", type=Path, help="corpus file")
    labels.add_argument("annotations", type=Path, help="annotation file (id + class)")
    labels.add_argument("--accounts", type=Path, help="account metadata file")
    labels.add_argument(
        "--dataset",
        required=True,
        help=(
            "provenance recorded on every label, e.g. "
            "indiana-bot-repository/varol-2017. Required: an evaluation that "
            "cannot name its annotation procedure is not reproducible."
        ),
    )
    labels.add_argument(
        "--numeric-convention",
        choices=["1_is_bot", "0_is_bot"],
        help="required when the annotation file uses 0 and 1 as classes",
    )
    labels.add_argument("--out", type=Path, help="write the labelled corpus here, as native JSON")

    train = subcommands.add_parser("train", help="fit a calibrated model and write its model card")
    train.add_argument("posts", type=Path, help="corpus file")
    train.add_argument("annotations", type=Path, help="annotation file (id + class)")
    train.add_argument("--accounts", type=Path, help="account metadata file")
    train.add_argument("--dataset", required=True, help="provenance for the labels")
    train.add_argument(
        "--numeric-convention",
        choices=["1_is_bot", "0_is_bot"],
        help="required when the annotation file uses 0 and 1 as classes",
    )
    train.add_argument(
        "--calibration",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="isotonic needs data; sigmoid (Platt) survives small samples",
    )
    train.add_argument("--folds", type=int, default=5, help="calibration and evaluation folds")
    train.add_argument(
        "--min-per-class",
        type=int,
        default=25,
        help="refuse to train below this many examples in either class (default: 25)",
    )
    train.add_argument("--card", type=Path, help="write the model card here, as JSON")
    train.add_argument(
        "--distribution",
        type=Path,
        help=(
            "write the corpus-level probability distribution here. Aggregate by "
            "design: per-account probabilities stay in the session."
        ),
    )

    docs = subcommands.add_parser("docs", help="regenerate the feature catalogue")
    docs.add_argument("--out", type=Path, default=Path("docs/features.md"))
    return parser


def _analyse(args: argparse.Namespace) -> int:
    """Run the pipeline and write whatever outputs were asked for."""
    adapter = NativeAdapter(
        platform=Platform(args.platform),
        assume_timezone=timezone_of(args.assume_timezone)
        if args.assume_timezone is not None
        else None,
        strict=args.strict,
    )
    loaded = (
        adapter.load_tables(args.posts, args.accounts)
        if args.accounts
        else adapter.load(args.posts)
    )
    report, features = build_report(
        loaded.corpus,
        coordination_config=CoordinationConfig(
            window=timedelta(minutes=args.window), min_edge_weight=args.min_edge_weight
        ),
        temporal_config=TemporalConfig(min_posts=args.min_posts),
        ingest_report=loaded.report,
        null_model_permutations=args.permutations,
        pseudonymise=not args.no_pseudonyms,
        pseudonym_salt=args.salt,
        include_examples=not args.no_examples,
        title=args.title,
    )

    written: list[str] = []
    if args.json:
        to_json(report, args.json)
        written.append(str(args.json))
    if args.html:
        to_html(report, args.html)
        written.append(str(args.html))
    if args.features:
        features.to_csv(args.features)
        written.append(str(args.features))

    report_summary = loaded.report
    print(
        f"loaded {report_summary.n_posts} posts from {report_summary.n_accounts} accounts"
        f" ({report_summary.n_skipped} rows skipped)"
    )
    for warning in report_summary.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    print(f"found {len(report.cards)} cluster(s) worth reporting")
    if args.permutations == 0 and report.cards:
        print(
            "  note: no null model was run, so that count has nothing to be "
            "compared against (--permutations 50)",
            file=sys.stderr,
        )
    for destination in written:
        print(f"wrote {destination}")
    if not written:
        print("no output written; pass --html, --json or --features", file=sys.stderr)
    return 0


def _labels(args: argparse.Namespace) -> int:
    """Join an annotation file to a corpus and report what it actually covers."""
    adapter = NativeAdapter()
    loaded = (
        adapter.load_tables(args.posts, args.accounts)
        if args.accounts
        else adapter.load(args.posts)
    )
    records = read_label_table(
        args.annotations,
        dataset=args.dataset,
        numeric_convention=args.numeric_convention,
    )
    labelled, coverage = attach_labels(loaded.corpus, records)

    print(
        f"{coverage.n_matched} of {coverage.n_labels} labels matched an account "
        f"({coverage.match_rate:.0%}); {coverage.n_unlabelled_accounts} accounts unlabelled"
    )
    for label, count in sorted(coverage.counts.items(), key=lambda item: item[0].value):
        print(f"  {label.value}: {count}")
    for warning in coverage.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    if args.out:
        write_corpus(labelled, args.out)
        print(f"wrote {args.out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    """Fit a calibrated model over the full feature matrix and report on it."""
    adapter = NativeAdapter()
    loaded = (
        adapter.load_tables(args.posts, args.accounts)
        if args.accounts
        else adapter.load(args.posts)
    )
    records = read_label_table(
        args.annotations, dataset=args.dataset, numeric_convention=args.numeric_convention
    )
    corpus, coverage = attach_labels(loaded.corpus, records)
    for warning in coverage.warnings:
        print(f"  warning: {warning}", file=sys.stderr)

    _, features = build_report(corpus, title="training run")
    model = AutomationEnsemble(
        EnsembleConfig(
            calibration=args.calibration,
            n_folds=args.folds,
            min_samples_per_class=args.min_per_class,
        )
    ).fit(features, corpus.labels)

    card = model.card
    assert card is not None
    evaluation = card.evaluation
    print(f"trained on {card.n_train} accounts; classes {dict(card.class_counts)}")
    print(
        f"  roc_auc {evaluation.roc_auc:.3f}  average_precision {evaluation.average_precision:.3f}"
    )
    print(
        f"  brier {evaluation.calibration.brier:.3f}  "
        f"calibration error {evaluation.calibration.expected_calibration_error:.3f}"
    )
    for name, scores in evaluation.per_class.items():
        print(f"  {name}: precision {scores['precision']:.2f} recall {scores['recall']:.2f}")
    for warning in card.warnings:
        print(f"  warning: {warning}", file=sys.stderr)

    if args.card:
        args.card.write_text(
            json.dumps(card.as_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"wrote {args.card}")
    if args.distribution:
        summary = summarise_probabilities(model.predict_proba(features))
        args.distribution.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {args.distribution}")
    return 0


def _docs(args: argparse.Namespace) -> int:
    """Regenerate the feature catalogue from the declared specs."""
    registry = FeatureRegistry.from_extractors(
        [AccountExtractor(), TemporalExtractor(), CoordinationExtractor()]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_catalogue(registry), encoding="utf-8")
    print(f"wrote {args.out} ({len(registry.specs)} features)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""
    args = _parser().parse_args(argv)
    if args.command == "analyse":
        return _analyse(args)
    if args.command == "labels":
        return _labels(args)
    if args.command == "train":
        return _train(args)
    return _docs(args)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
