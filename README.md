# synthwatch

Platform-agnostic toolkit for measuring **automation and AI-generated text in
political conversation**. Academic / portfolio project.

> `synthwatch` estimates *how much* of a conversation looks automated or
> synthetic, and *which clusters* behave in a coordinated way. It does not
> decide that a given person is a bot, and it is not built to. See
> [`ETHICS.md`](ETHICS.md).

## Layers

| Layer | Module | Responsibility |
| --- | --- | --- |
| 1 | `synthwatch.ingest` | Adapters (native CSV/JSON, Reddit, Bluesky, Mastodon) normalising into one internal schema |
| 2 | `synthwatch.detect` | Feature extractors `text`, `account`, `temporal`, `coordination`, plus a calibrated `ensemble` |
| 3 | `synthwatch.report` | Aggregate metrics, cluster cards, JSON/HTML export |

## Status

Runs end to end, and has been trained. Against 37,438 labelled accounts the
calibrated ensemble reaches **ROC AUC 0.872** with an expected calibration
error of **0.005** — of the accounts it calls 25% likely, 25% are labelled
bots, and that holds across every bin. Read it as a floor: that dataset carries
no posts, so 15 of the 22 features were unmeasurable and the model card says so.
See [`docs/datasets.md`](docs/datasets.md).

- [x] Internal schema (`Account`, `Post`, `LabelRecord`, `Corpus`)
- [x] Feature declaration contract (`FeatureSpec`, enforced by tests)
- [x] Adapter contract (`Adapter`, `IngestReport`)
- [x] `detect.coordination` — SimHash near-duplicates, banded candidate
      generation, Louvain communities, permutation null model, seven
      account-level features
- [x] `ingest.native` — CSV, TSV, JSON and JSONL in the internal schema, with
      column aliases for the usual archive exports, a corpus cache writer, and
      a load report that counts every dropped row and why
- [x] `detect.temporal` — circadian shape, quiet-window share, inter-arrival
      entropy, self-relative burst detection; every feature invariant to the
      account's timezone
- [x] `detect.account` — age, handle shape, network ratios, profile
      completeness, posting rate, dormancy before the first observed post
- [x] `report` + CLI — feature distributions with their coverage, pseudonymised
      cluster cards, self-contained HTML and JSON export
- [x] `ingest.labelled` — annotation files and Twitter Information Operations
      Archive takedowns, with a coverage report that names the conditions
      under which a trained model would be untrustworthy
- [x] `detect.ensemble` — calibrated gradient boosting over all 22 features,
      SHAP attributions, and a model card that records how it was trained
- [x] `ingest.inspect` — profile an unknown file before loading it: which
      columns map, which are missing, what a trial load would skip and why
- [ ] `detect.text` (needs the `text` extra: transformers + torch)
- [ ] ingest adapters for Reddit, Bluesky and Mastodon

## Running it

```bash
synthwatch analyse posts.csv --accounts accounts.csv \
    --html report.html --json report.json --permutations 50
```

Out comes a single self-contained HTML file: no external stylesheet, no script,
no network request when it is opened. It leads with its own caveats, reports
each feature's distribution next to that feature's known limitation and its
coverage, and describes clusters rather than accounts.

Three things the report layer enforces rather than merely documents:

- **Account identifiers are pseudonymised on the way out.** Stable within a
  report, salted, and the mapping is never written to disk. It is a speed bump
  against casual misuse, not anonymisation — a cluster described here is
  re-identifiable by anyone with platform access, and the report says so.
- **Per-account values never enter the document.** `build_report()` returns the
  feature matrix to the caller instead of embedding it; `--features` writes it
  to a file that stays with the analyst.
- **Every figure travels with the parameters that produced it.** If no null
  model was run, the report says the cluster count has nothing to be compared
  against.

## Meeting a dataset for the first time

```bash
synthwatch inspect whatever_you_downloaded.csv
```

Samples a couple of thousand rows and reports which columns the schema
recognises, which would land in `extra`, which required field has nothing
feeding it, and exactly what a load would skip and why. It never reads the
whole file, so pointing it at a hundred-gigabyte archive returns immediately.

Run against the published archive of X's information operations, it turns up
two things the documentation does not mention: profile creation dates written
as `7/3/2018`, which is either March or July, and account ids a spreadsheet
round-trip has rewritten as `1.01421E+18` — three of them in a 352-row sample.
Both are refused rather than guessed at, and counted rather than dropped.

See [`docs/datasets.md`](docs/datasets.md) for where to get real data and what
is wrong with each source.

## Training a model

```bash
synthwatch train posts.csv labels.dat \
    --dataset indiana-bot-repository/varol-2017 --card model_card.json
```

Real output, against 37,438 labelled accounts:

```
trained on 37438 accounts; classes {0: 25013, 1: 12425}
  roc_auc 0.872  average_precision 0.800
  brier 0.130  calibration error 0.005
  negative: precision 0.83 recall 0.91
  positive: precision 0.77 recall 0.64
  warning: 15 feature(s) were unmeasurable for every labelled account and
           were dropped. The model answers from fewer signals than the
           pipeline produces.
```

That warning is the honest part: the dataset carries no posts, so every
temporal and coordination feature was dropped and this is what the profile
features alone can do.

Four things the model refuses to do:

- **It never imputes a missing value.** The extractors work to distinguish "we
  measured zero" from "we could not measure", and an imputer erases that in one
  line. `HistGradientBoostingClassifier` sends missing values down their own
  branch, so absence is a signal rather than a hole someone filled with a median.
- **It has no `predict`.** Only calibrated probabilities. Turning one into a
  decision needs a threshold, and a threshold states how much worse a false
  accusation is than a missed bot — a question for whoever is accountable for
  the consequences.
- **It never computes accuracy.** On a label set where one class holds 80% of
  the rows, accuracy measures that class. Per-class precision and recall, Brier
  and expected calibration error are what the card carries.
- **It refuses to train** on a single-class label set, or on fewer examples than
  `--min-per-class`, and it says so when the model turns out to be constant.

## Labelled data, and what it will not tell you

```bash
synthwatch labels accounts.json labels.tsv --dataset airt-ml/twitter-human-bots
```

```
412 of 1500 labels matched an account (27%); 88 accounts unlabelled
  automated: 190
  organic: 222
  warning: only 412 of 1500 labels matched an account in this corpus...
```

The loaders refuse to guess in two places that would otherwise be invisible.
A numeric annotation file (`0`/`1`) is rejected unless the caller declares which
way round it runs — both conventions exist in published datasets, and choosing
one silently inverts a training set. An unrecognised class string raises rather
than quietly becoming `UNKNOWN`, because a dropped class is a shrunken training
set nobody notices.

The Information Operations Archive gets its own adapter, and every account in a
takedown is labelled `INFO_OPERATION` with `method=PLATFORM_ENFORCEMENT` —
**not** `AUTOMATED`. Many of those accounts were run by people, by hand, full
time. The load warns that the takedown carries a single class: a model trained
on it plus a control group collected some other way learns to tell the two
*collections* apart and reports an excellent score for doing it.

## Loading data

```python
from pathlib import Path
from synthwatch.ingest import NativeAdapter

result = NativeAdapter().load_tables(Path("posts.csv"), Path("accounts.csv"))
result.report.as_dict()  # rows read, rows skipped, and the reason for each
corpus = result.corpus
```

Column names from the Twitter Information Operations Archive and Pushshift-style
dumps are recognised out of the box; anything else maps through
`post_aliases=` / `account_aliases=`. Columns the schema has no field for are
kept in `extra` rather than dropped.

A timestamp without a UTC offset is **refused**, not assumed to be UTC — pass
`assume_timezone=` to state what the file actually contains. Guessing here
rotates every circadian and inter-arrival feature downstream, and the wrong
answer looks exactly as plausible as the right one.

## Timezones, and why the temporal features do not need them

A corpus almost never says which timezone an account lives in, so
`detect.temporal` is built so that no feature can depend on it. Instead of
asking whether an account posts at 3am local time, it asks how concentrated
posting is across the 24 hours whatever they are called, and how much the
account posts during *its own* quietest six-hour stretch. Rotating a whole
timeline leaves both unchanged — a property the test suite checks directly.
The exception, documented in the module: half-hour offsets such as India's
move posts across hour boundaries instead of rotating them, so there the
invariance is approximate rather than exact.

## Coordination analysis at a glance

```python
from synthwatch.detect.coordination import CoordinationConfig, detect_coordination

result = detect_coordination(
    corpus,
    CoordinationConfig(window=timedelta(minutes=15), min_edge_weight=2),
    null_model_permutations=100,
)
result.as_dict()["clusters"]  # cluster cards, never per-account verdicts
result.as_dict()["null_model"]  # observed edges vs. time-randomised corpora
```

Two things the module insists on: the parameters that produced a result travel
with it (`config` is part of every export), and the cluster count is compared
against what the same corpus produces by chance before it is reported.
`window_sensitivity()` re-runs the analysis across several windows, because a
cluster that only exists at one window is not a finding.

## Development

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Python 3.11+. Core dependencies stay light (pydantic, pandas, numpy,
networkx). The language model behind perplexity and burstiness lives in the
`text` extra, so ingest, account, temporal and coordination analysis all run
without a GPU or a model download.

## Documentation

- [`docs/schema.md`](docs/schema.md) — the internal schema and the decisions behind it
- [`docs/features.md`](docs/features.md) — every feature with its rationale and its known limitation
- [`docs/datasets.md`](docs/datasets.md) — where to get real data, and the defect in each source
- [`ETHICS.md`](ETHICS.md) — scope, refusals, and how results should be read

## License

MIT.
