"""Gradient boosting over every feature family, calibrated, with SHAP.

This is where the library stops describing and starts estimating, so it is also
where it is easiest to produce something confident and wrong. Four decisions
exist to make that harder.

**Missing values are never imputed.** The extractors work hard to distinguish
"we measured this and it was zero" from "we could not measure this at all", and
an imputer erases that distinction in one line. ``HistGradientBoostingClassifier``
routes missing values down their own branch at every split, so absence is a
signal the model can learn from rather than a hole someone filled with a median.

**Calibration is fit on held-out folds.** A gradient booster's raw scores are
not probabilities; calibrating them on the data they were fit to produces a
reliability curve that looks perfect and means nothing.

**There is no ``predict``.** The class emits calibrated probabilities and
nothing else. Turning a probability into a decision requires a threshold, and a
threshold is a statement about how much worse a false accusation is than a
missed bot. That is a policy question belonging to whoever is accountable for
the consequences, not a default in a library.

**Evaluation reports per-class performance and calibration quality.** Accuracy
on an imbalanced label set measures the majority class; this module does not
compute it.

Known limitations
-----------------

* **The uncertainty band is model instability, not statistical uncertainty.**
  It is the spread across the calibration folds: how much the answer depends on
  which slice of the training data a model saw. A narrow band means the folds
  agreed, not that the estimate is correct.
* **Cross-validation over accounts leaks context.** Accounts from one campaign
  land in both train and test, so a fold score flatters the model relative to
  what it would do on a campaign it has never seen. Grouped evaluation needs
  group labels the public datasets do not ship.
* **A model trained on one platform and era transfers badly.** The features are
  platform-agnostic; the thresholds the model learns over them are not.
* **SHAP explains the ranking model, not the calibrated one.** Calibration is a
  monotonic transform applied on top, so attributions keep their order and
  their relative size but are expressed in the uncalibrated model's units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from synthwatch import __version__
from synthwatch.models import LabelRecord
from synthwatch.types import AccountId, Label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "AutomationEnsemble",
    "CalibrationReport",
    "EnsembleConfig",
    "EvaluationReport",
    "InsufficientLabelsError",
    "ModelCard",
    "TrainingTask",
    "build_training_frame",
    "expected_calibration_error",
]

CalibrationMethod = Literal["isotonic", "sigmoid"]

RELIABILITY_BINS = 10
"""Bins for the reliability curve and the expected calibration error."""

MIN_CLASSES = 2
"""Below this, a training set describes a collection rather than a behaviour."""

METRIC_THRESHOLD = 0.5
"""Cut used only to compute per-class precision and recall.

It is emphatically not a recommended operating point. Per-class metrics need
some threshold to exist at all, so this one is fixed and disclosed rather than
tuned; choosing where to actually cut is a decision this library does not make.
"""

IMBALANCE_WARNING_SHARE = 0.2
"""Below this minority share, the probabilities get pulled towards the base rate."""

POOR_CALIBRATION_ECE = 0.1
"""Above this expected calibration error, the probabilities do not mean what they say."""

DEGENERATE_RANGE = 1e-6
"""Below this spread in the out-of-fold probabilities, the model is a constant.

A constant predictor still produces a reasonable-looking Brier score at a
balanced base rate, so the condition has to be detected directly rather than
inferred from a metric.
"""


class InsufficientLabelsError(ValueError):
    """Raised when a label set cannot support a trustworthy model."""


def _require_sklearn() -> None:
    """Fail early, with an error that says how to fix it."""
    try:
        import sklearn  # noqa: F401, PLC0415
    except ImportError as error:  # pragma: no cover - exercised by the extra being absent
        msg = (
            'the ensemble needs scikit-learn. Install the extra: pip install "synthwatch[ensemble]"'
        )
        raise ImportError(msg) from error


@dataclass(frozen=True, slots=True)
class TrainingTask:
    """Which classes the model is being asked to separate.

    Pooling classes is a modelling choice with consequences, so it has to be
    written down rather than assumed. ``AUTOMATED`` and ``INFO_OPERATION`` are
    *not* the same thing -- many accounts in a takedown were operated by hand --
    and a task that merges them is answering a different question from one that
    does not.

    Attributes:
        name: What the resulting probability means, in one word.
        positive: Labels treated as the positive class.
        negative: Labels treated as the negative class.
    """

    name: str = "automation"
    positive: frozenset[Label] = frozenset({Label.AUTOMATED})
    negative: frozenset[Label] = frozenset({Label.ORGANIC})

    def __post_init__(self) -> None:
        """Reject a task whose classes overlap or are empty."""
        if not self.positive or not self.negative:
            msg = "a task needs at least one label on each side"
            raise ValueError(msg)
        overlap = self.positive & self.negative
        if overlap:
            names = ", ".join(sorted(label.value for label in overlap))
            msg = f"labels on both sides of the task: {names}"
            raise ValueError(msg)

    def side_of(self, label: Label) -> int | None:
        """``1``, ``0``, or ``None`` for a label this task does not cover."""
        if label in self.positive:
            return 1
        if label in self.negative:
            return 0
        return None

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the model card."""
        return {
            "name": self.name,
            "positive": sorted(label.value for label in self.positive),
            "negative": sorted(label.value for label in self.negative),
        }


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Parameters of the model and of how it is evaluated.

    Attributes:
        task: Which classes to separate.
        calibration: ``"isotonic"`` is flexible and needs data;
            ``"sigmoid"`` (Platt) assumes a shape and survives small samples.
        n_folds: Folds used for both calibration and out-of-fold evaluation.
        max_iter: Boosting iterations.
        learning_rate: Boosting learning rate.
        max_leaf_nodes: Tree size.
        min_samples_leaf: Minimum rows in a leaf. scikit-learn defaults to
            20, which is wrong for this domain: labelled datasets here run
            to a few hundred accounts, folds are smaller still, and a fold
            of 32 rows cannot be split at all when both leaves need 20. The
            tree collapses to a single leaf, calibration maps everything to
            the base rate, and the result is a model that reports a
            respectable Brier score while having learned nothing.
        min_samples_per_class: Below this in either class, training is refused.
            Not a statistical threshold so much as a refusal to produce a
            confident-looking model from thirty examples.
        seed: Fixes every random choice, so a model card is reproducible.
    """

    task: TrainingTask = field(default_factory=TrainingTask)
    calibration: CalibrationMethod = "isotonic"
    n_folds: int = 5
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 5
    min_samples_per_class: int = 25
    seed: int = 20240301

    def __post_init__(self) -> None:
        """Reject parameter combinations that would silently misbehave."""
        if self.n_folds < MIN_CLASSES:
            msg = f"n_folds must be at least {MIN_CLASSES}; calibration needs held-out data"
            raise ValueError(msg)
        if self.calibration not in ("isotonic", "sigmoid"):
            msg = f"unknown calibration {self.calibration!r}; use 'isotonic' or 'sigmoid'"
            raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the model card."""
        return {
            "task": self.task.as_dict(),
            "calibration": self.calibration,
            "n_folds": self.n_folds,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_per_class": self.min_samples_per_class,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------
# Assembling a training set
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingFrame:
    """A feature matrix and its labels, with what was left out and why."""

    features: pd.DataFrame
    target: pd.Series
    excluded: Mapping[str, int]

    @property
    def class_counts(self) -> dict[int, int]:
        """Rows per class."""
        counts = self.target.value_counts()
        return {int(str(value)): int(count) for value, count in counts.items()}


def build_training_frame(
    features: pd.DataFrame,
    labels: Sequence[LabelRecord],
    task: TrainingTask | None = None,
) -> TrainingFrame:
    """Join a feature matrix to its labels, counting every exclusion.

    Rows are dropped for three reasons, each counted separately: the account has
    no label, its label belongs to neither side of the task, or every one of its
    features is missing. An all-missing row carries no information and would
    only teach the model the base rate.
    """
    task = task or TrainingTask()
    by_account: dict[AccountId, Label] = {record.account_id: record.label for record in labels}
    excluded: dict[str, int] = {"unlabelled": 0, "outside_task": 0, "no_measurable_features": 0}

    rows: list[AccountId] = []
    targets: list[int] = []
    for account_id in features.index:
        label = by_account.get(str(account_id))
        if label is None:
            excluded["unlabelled"] += 1
            continue
        side = task.side_of(label)
        if side is None:
            excluded["outside_task"] += 1
            continue
        if bool(features.loc[account_id].isna().all()):
            excluded["no_measurable_features"] += 1
            continue
        rows.append(str(account_id))
        targets.append(side)

    return TrainingFrame(
        features=features.loc[rows],
        target=pd.Series(targets, index=pd.Index(rows, name="account_id"), name=task.name),
        excluded=excluded,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def expected_calibration_error(
    truth: NDArray[np.int_], probability: NDArray[np.float64], *, bins: int = RELIABILITY_BINS
) -> float:
    """Average gap between predicted confidence and observed frequency.

    Of everything the model said was 70% likely, how much of it turned out to
    be positive? The answer to that is what a probability from this library is
    claiming, so it is the number worth reporting -- more than any ranking
    metric, which says nothing about whether ``0.7`` means seven in ten.
    """
    if len(truth) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in pairwise(edges):
        in_bin = (probability > lower) & (probability <= upper)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        total += weight * abs(truth[in_bin].mean() - probability[in_bin].mean())
    return float(total)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Whether the probabilities mean what they say."""

    brier: float
    expected_calibration_error: float
    reliability: tuple[tuple[float, float, int], ...]
    """``(mean predicted, observed frequency, count)`` per populated bin."""

    def as_dict(self) -> dict[str, object]:
        """Serialisable view."""
        return {
            "brier": round(self.brier, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "reliability": [
                {"predicted": round(p, 3), "observed": round(o, 3), "n": n}
                for p, o, n in self.reliability
            ],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Out-of-fold performance, reported per class.

    Accuracy is deliberately absent. On a label set where one class holds 90% of
    the rows, it is a measure of that class and nothing else.
    """

    n_samples: int
    base_rate: float
    roc_auc: float | None
    average_precision: float | None
    per_class: Mapping[str, Mapping[str, float]]
    calibration: CalibrationReport

    def as_dict(self) -> dict[str, object]:
        """Serialisable view for the model card."""
        return {
            "n_samples": self.n_samples,
            "base_rate": round(self.base_rate, 4),
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 4),
            "average_precision": (
                None if self.average_precision is None else round(self.average_precision, 4)
            ),
            "per_class": {
                name: {metric: round(value, 4) for metric, value in scores.items()}
                for name, scores in self.per_class.items()
            },
            "calibration": self.calibration.as_dict(),
        }


def _reliability_curve(
    truth: NDArray[np.int_], probability: NDArray[np.float64], *, bins: int = RELIABILITY_BINS
) -> tuple[tuple[float, float, int], ...]:
    """Mean prediction against observed frequency, per populated bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[tuple[float, float, int]] = []
    for lower, upper in pairwise(edges):
        in_bin = (probability > lower) & (probability <= upper)
        if not in_bin.any():
            continue
        curve.append(
            (float(probability[in_bin].mean()), float(truth[in_bin].mean()), int(in_bin.sum()))
        )
    return tuple(curve)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Everything needed to judge whether a model's output means anything.

    A probability from this library that arrives without its card is not
    interpretable, so the card is part of the object rather than something a
    conscientious user is expected to write afterwards.
    """

    task: Mapping[str, object]
    config: Mapping[str, object]
    features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    n_train: int
    class_counts: Mapping[int, int]
    excluded: Mapping[str, int]
    datasets: tuple[str, ...]
    label_methods: tuple[str, ...]
    evaluation: EvaluationReport
    warnings: tuple[str, ...]
    library_version: str = __version__
    trained_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, suitable for embedding in a report."""
        return {
            "library_version": self.library_version,
            "trained_at": self.trained_at.isoformat(),
            "task": dict(self.task),
            "config": dict(self.config),
            "features": list(self.features),
            "dropped_features": list(self.dropped_features),
            "n_train": self.n_train,
            "class_counts": {str(k): v for k, v in self.class_counts.items()},
            "excluded": dict(self.excluded),
            "datasets": list(self.datasets),
            "label_methods": list(self.label_methods),
            "evaluation": self.evaluation.as_dict(),
            "warnings": list(self.warnings),
        }


class AutomationEnsemble:
    """Calibrated gradient boosting over the full feature matrix.

    The model exposes :meth:`predict_proba` and no ``predict``. Choosing where
    to cut a probability is a decision about the relative cost of a false
    accusation and a missed account, and this library is not the right place
    for that decision to be made silently.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.card: ModelCard | None = None
        self._calibrated: Any = None
        self._explainer_model: Any = None
        self._features: tuple[str, ...] = ()
        self._degenerate = False

    # -- training --------------------------------------------------------

    def _estimator(self) -> Any:  # noqa: ANN401 - a sklearn estimator, unimportable at module scope
        """A booster that treats a missing value as a branch, not a hole."""
        _require_sklearn()
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

        return HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.seed,
        )

    def fit(self, features: pd.DataFrame, labels: Sequence[LabelRecord]) -> AutomationEnsemble:
        """Train on ``features`` joined to ``labels``.

        Raises:
            InsufficientLabelsError: If either class has fewer rows than
                ``min_samples_per_class``, or only one class is present.
        """
        _require_sklearn()
        from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        frame = build_training_frame(features, labels, self.config.task)
        counts = frame.class_counts
        self._check_trainable(counts)

        # A feature that was never measurable in this corpus cannot contribute,
        # and an all-missing column has no bins for the booster to split on.
        # Dropping it is recorded on the card rather than done quietly: it means
        # the model is answering from fewer signals than the pipeline produces.
        usable = frame.features.dropna(axis=1, how="all")
        dropped = tuple(str(name) for name in frame.features.columns if name not in usable.columns)
        if usable.empty or not len(usable.columns):
            msg = (
                "no feature was measurable for any labelled account; there is nothing to train on."
            )
            raise InsufficientLabelsError(msg)

        matrix = usable.to_numpy(dtype=float)
        target = frame.target.to_numpy(dtype=int)
        folds = min(self.config.n_folds, *counts.values())
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.seed)

        self._calibrated = CalibratedClassifierCV(
            estimator=self._estimator(), method=self.config.calibration, cv=splitter, ensemble=True
        ).fit(matrix, target)
        # A second, uncalibrated fit on everything, used only for attribution.
        # Calibration is monotonic, so this preserves the ordering and the
        # relative size of every contribution while staying a tree model SHAP
        # can read directly.
        self._explainer_model = self._estimator().fit(matrix, target)
        self._features = tuple(str(name) for name in usable.columns)

        evaluation = self._evaluate(matrix, target, folds)
        self.card = ModelCard(
            task=self.config.task.as_dict(),
            config=self.config.as_dict(),
            features=self._features,
            dropped_features=dropped,
            n_train=len(target),
            class_counts=counts,
            excluded=frame.excluded,
            datasets=tuple(sorted({record.dataset for record in labels})),
            label_methods=tuple(sorted({record.method.value for record in labels})),
            evaluation=evaluation,
            warnings=self._warnings(counts, evaluation, labels, dropped),
        )
        return self

    def _check_trainable(self, counts: Mapping[int, int]) -> None:
        """Refuse a label set that cannot support a trustworthy model."""
        if len(counts) < MIN_CLASSES:
            present = ", ".join(str(key) for key in counts) or "none"
            msg = (
                f"only one class present ({present}). A single-class training set "
                "produces a model that has learned the collection, not the behaviour."
            )
            raise InsufficientLabelsError(msg)
        smallest = min(counts.values())
        if smallest < self.config.min_samples_per_class:
            msg = (
                f"smallest class has {smallest} rows, below min_samples_per_class="
                f"{self.config.min_samples_per_class}. Lower it deliberately if you "
                "mean to, but a model fitted here will look confident and be noise."
            )
            raise InsufficientLabelsError(msg)

    def _evaluate(
        self, matrix: NDArray[np.float64], target: NDArray[np.int_], folds: int
    ) -> EvaluationReport:
        """Score the pipeline out of fold, including its calibration."""
        from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
        from sklearn.metrics import (  # noqa: PLC0415
            average_precision_score,
            brier_score_loss,
            precision_recall_fscore_support,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: PLC0415

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.seed)
        pipeline = CalibratedClassifierCV(
            estimator=self._estimator(),
            method=self.config.calibration,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.seed + 1),
            ensemble=True,
        )
        out_of_fold = cross_val_predict(
            pipeline, matrix, target, cv=splitter, method="predict_proba"
        )[:, 1]
        # A constant predictor still scores a respectable Brier at a balanced
        # base rate, so the condition is detected here rather than inferred
        # from a metric that will not reveal it.
        self._degenerate = bool(np.ptp(out_of_fold) < DEGENERATE_RANGE)

        predicted_class = (out_of_fold >= METRIC_THRESHOLD).astype(int)
        precision, recall, f_score, support = precision_recall_fscore_support(
            target, predicted_class, labels=[0, 1], zero_division=0
        )
        per_class = {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f_score[index]),
                "support": float(support[index]),
            }
            for index, name in enumerate(("negative", "positive"))
        }
        return EvaluationReport(
            n_samples=len(target),
            base_rate=float(target.mean()),
            roc_auc=float(roc_auc_score(target, out_of_fold)),
            average_precision=float(average_precision_score(target, out_of_fold)),
            per_class=per_class,
            calibration=CalibrationReport(
                brier=float(brier_score_loss(target, out_of_fold)),
                expected_calibration_error=expected_calibration_error(target, out_of_fold),
                reliability=_reliability_curve(target, out_of_fold),
            ),
        )

    def _warnings(
        self,
        counts: Mapping[int, int],
        evaluation: EvaluationReport,
        labels: Sequence[LabelRecord],
        dropped: Sequence[str],
    ) -> tuple[str, ...]:
        """Conditions that make this model's output harder to trust."""
        warnings: list[str] = []
        if self._degenerate:
            warnings.append(
                "the model produced an essentially constant probability: it "
                "separated nothing. Usually there is too little data for the "
                "folds, or min_samples_leaf is larger than a fold can support. "
                "Do not report these numbers."
            )
        if dropped:
            warnings.append(
                f"{len(dropped)} feature(s) were unmeasurable for every labelled "
                f"account and were dropped ({', '.join(sorted(dropped))}). The model "
                "answers from fewer signals than the pipeline produces."
            )
        minority = min(counts.values()) / sum(counts.values())
        if minority < IMBALANCE_WARNING_SHARE:
            warnings.append(
                f"the minority class holds {minority:.0%} of the training set; read "
                "precision and recall per class, and expect the probabilities to be "
                "pulled towards the base rate."
            )
        if evaluation.calibration.expected_calibration_error > POOR_CALIBRATION_ECE:
            warnings.append(
                "expected calibration error above 0.1: these probabilities do not yet "
                "mean what they say. Try the other calibration method, or more data."
            )
        datasets = {record.dataset for record in labels}
        if len(datasets) > 1:
            warnings.append(
                f"trained on {len(datasets)} datasets with different annotation "
                "procedures; report per-dataset performance before pooling the claim."
            )
        warnings.append(
            "cross-validation here splits accounts, not campaigns, so accounts from "
            "one operation appear in both training and test. Scores are optimistic "
            "relative to an unseen campaign."
        )
        return tuple(warnings)

    # -- inference -------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calibrated probabilities, with the spread across calibration folds.

        Returns:
            A frame indexed by account id with ``probability`` (the calibrated
            estimate), ``low`` and ``high`` (the minimum and maximum across the
            per-fold calibrated models), and ``spread``.

        Note:
            The band measures how much the answer depends on which slice of the
            training data a model saw. It is not a confidence interval for the
            true probability, and a narrow band is agreement, not correctness.
        """
        self._require_fitted()
        aligned = self._align(features)
        matrix = aligned.to_numpy(dtype=float)
        per_fold = np.vstack(
            [
                model.predict_proba(matrix)[:, 1]
                for model in self._calibrated.calibrated_classifiers_
            ]
        )
        return pd.DataFrame(
            {
                "probability": self._calibrated.predict_proba(matrix)[:, 1],
                "low": per_fold.min(axis=0),
                "high": per_fold.max(axis=0),
                "spread": per_fold.max(axis=0) - per_fold.min(axis=0),
            },
            index=aligned.index,
        )

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-account SHAP values, in the uncalibrated model's units.

        Returned to the caller rather than written into a report: these are
        per-account numbers, and the report layer publishes aggregates.

        Raises:
            ImportError: If SHAP is not installed.
        """
        self._require_fitted()
        try:
            import shap  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - exercised by the extra being absent
            msg = 'explanations need SHAP. Install: pip install "synthwatch[ensemble]"'
            raise ImportError(msg) from error

        aligned = self._align(features)
        values = shap.TreeExplainer(self._explainer_model).shap_values(
            aligned.to_numpy(dtype=float)
        )
        array = np.asarray(values)
        # Some SHAP versions return one matrix per class; the positive one is
        # what a probability of automation is about.
        per_class_dimensions = 3
        if array.ndim == per_class_dimensions:
            array = array[:, :, -1]
        return pd.DataFrame(array, index=aligned.index, columns=list(self._features))

    def global_importance(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        """Mean absolute SHAP value per feature, most influential first."""
        contributions = self.explain(features).abs().mean()
        return sorted(
            ((str(name), float(value)) for name, value in contributions.items()),
            key=lambda item: item[1],
            reverse=True,
        )

    # -- helpers ---------------------------------------------------------

    def _require_fitted(self) -> None:
        """Raise unless :meth:`fit` has been called."""
        if self._calibrated is None:
            msg = "this ensemble has not been fitted"
            raise RuntimeError(msg)

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to the training layout, refusing to invent any.

        A feature absent at inference time is a different pipeline from the one
        that was trained, and quietly filling it with NaN would let a model
        built on twenty-two signals answer from nineteen without saying so.
        """
        missing = [name for name in self._features if name not in features.columns]
        if missing:
            msg = (
                f"features missing at inference time: {', '.join(missing)}. The model "
                "was trained on a different pipeline than the one being applied."
            )
            raise ValueError(msg)
        return features[list(self._features)]


def summarise_probabilities(probabilities: pd.DataFrame, *, bins: int = 10) -> dict[str, object]:
    """Corpus-level view of a probability column, for the report layer.

    Aggregate on purpose: a histogram and a few quantiles describe a corpus
    without publishing what the model thinks about any one account.
    """
    column = probabilities["probability"].dropna()
    if column.empty:
        return {"n": 0, "histogram": [], "quantiles": {}, "mean_spread": None}
    counts, edges = np.histogram(column.to_numpy(dtype=float), bins=bins, range=(0.0, 1.0))
    return {
        "n": int(column.size),
        "histogram": [
            {"from": round(float(low), 2), "to": round(float(high), 2), "n": int(count)}
            for low, high, count in zip(edges[:-1], edges[1:], counts, strict=True)
        ],
        "quantiles": {
            "p10": round(float(column.quantile(0.1)), 4),
            "median": round(float(column.median()), 4),
            "p90": round(float(column.quantile(0.9)), 4),
        },
        "mean_spread": round(float(probabilities["spread"].mean()), 4),
        "caveat": (
            "A distribution of calibrated probabilities is not a count of automated "
            "accounts. Converting it into one requires a threshold, and a threshold "
            "is a decision about the cost of being wrong in each direction."
        ),
    }
