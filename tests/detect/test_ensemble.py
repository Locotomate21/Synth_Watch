"""Tests for the calibrated ensemble.

The fixtures plant a signal the model should find, and most of the assertions
are about what it must refuse to do with it: no threshold, no imputation, no
silent training on a label set too small or too one-sided to support a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from synthwatch.cli import main
from synthwatch.detect.ensemble import (
    AutomationEnsemble,
    EnsembleConfig,
    InsufficientLabelsError,
    ModelCard,
    TrainingTask,
    build_training_frame,
    expected_calibration_error,
    summarise_probabilities,
)
from synthwatch.ingest.native import write_corpus
from synthwatch.models import LabelRecord, build_corpus
from synthwatch.types import Label, LabelMethod, Platform
from tests.conftest import make_account, make_post

FEATURES = ("acct_age_days", "temp_circadian_entropy", "coord_partner_count", "acct_handle_entropy")


def synthetic(
    n_per_class: int = 120, *, missing_rate: float = 0.15, seed: int = 11
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    """A feature matrix with a real but noisy signal, and holes in it.

    The holes matter: they are how the real extractors report "not measurable",
    and a model that cannot cope with them is not usable on real data.
    """
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    ids: list[str] = []
    labels: list[LabelRecord] = []

    for index in range(n_per_class * 2):
        automated = index >= n_per_class
        account_id = f"{'bot' if automated else 'human'}{index:04d}"
        shift = 1.4 if automated else -1.4
        rows.append(
            [
                float(rng.normal(loc=-shift, scale=1.2)),
                float(rng.normal(loc=shift * 0.6, scale=1.0)),
                float(rng.normal(loc=shift, scale=1.3)),
                float(rng.normal(loc=0.0, scale=1.0)),  # pure noise
            ]
        )
        ids.append(account_id)
        labels.append(
            LabelRecord(
                account_id=account_id,
                platform=Platform.GENERIC,
                label=Label.AUTOMATED if automated else Label.ORGANIC,
                dataset="synthetic/planted",
                method=LabelMethod.HUMAN_ANNOTATION,
            )
        )

    frame = pd.DataFrame(rows, index=pd.Index(ids, name="account_id"), columns=list(FEATURES))
    holes = rng.random(frame.shape) < missing_rate
    return frame.mask(holes), labels


@pytest.fixture(scope="module")
def trained() -> tuple[AutomationEnsemble, pd.DataFrame, list[LabelRecord]]:
    features, labels = synthetic()
    model = AutomationEnsemble(EnsembleConfig(max_iter=60, n_folds=4)).fit(features, labels)
    return model, features, labels


# -------------------------------------------------------------------- tasks


class TestTrainingTask:
    def test_the_default_task_separates_automated_from_organic(self):
        task = TrainingTask()
        assert task.side_of(Label.AUTOMATED) == 1
        assert task.side_of(Label.ORGANIC) == 0

    def test_a_label_outside_the_task_has_no_side(self):
        # INFO_OPERATION is not a synonym for AUTOMATED, so the default task
        # does not quietly absorb it.
        assert TrainingTask().side_of(Label.INFO_OPERATION) is None

    def test_pooling_classes_is_possible_but_must_be_written_down(self):
        task = TrainingTask(
            name="inauthentic",
            positive=frozenset({Label.AUTOMATED, Label.INFO_OPERATION}),
            negative=frozenset({Label.ORGANIC}),
        )
        assert task.side_of(Label.INFO_OPERATION) == 1
        assert task.as_dict()["positive"] == ["automated", "info_operation"]

    def test_overlapping_sides_are_refused(self):
        with pytest.raises(ValueError, match="both sides"):
            TrainingTask(
                positive=frozenset({Label.AUTOMATED}),
                negative=frozenset({Label.AUTOMATED, Label.ORGANIC}),
            )

    def test_an_empty_side_is_refused(self):
        with pytest.raises(ValueError, match="at least one label"):
            TrainingTask(positive=frozenset())


class TestConfig:
    def test_one_fold_cannot_calibrate(self):
        with pytest.raises(ValueError, match="n_folds"):
            EnsembleConfig(n_folds=1)

    def test_an_unknown_calibration_method_is_refused(self):
        with pytest.raises(ValueError, match="unknown calibration"):
            EnsembleConfig(calibration="platt")  # type: ignore[arg-type]

    def test_config_is_serialisable_for_the_card(self):
        exported = EnsembleConfig().as_dict()
        assert exported["calibration"] == "isotonic"
        assert cast("dict[str, Any]", exported["task"])["positive"] == ["automated"]


# ------------------------------------------------------------- training set


class TestTrainingFrame:
    def test_it_joins_features_to_labels(self):
        features, labels = synthetic(n_per_class=30)
        frame = build_training_frame(features, labels)
        assert len(frame.target) == 60
        assert frame.class_counts == {0: 30, 1: 30}

    def test_unlabelled_accounts_are_counted_not_silently_dropped(self):
        features, labels = synthetic(n_per_class=30)
        frame = build_training_frame(features, labels[:40])
        assert frame.excluded["unlabelled"] == 20

    def test_labels_outside_the_task_are_counted_separately(self):
        features, labels = synthetic(n_per_class=30)
        recast = [
            record.model_copy(update={"label": Label.INFO_OPERATION}) if index < 10 else record
            for index, record in enumerate(labels)
        ]
        frame = build_training_frame(features, recast)
        assert frame.excluded["outside_task"] == 10

    def test_rows_with_nothing_measurable_are_excluded(self):
        # An all-NaN row can only teach the model the base rate.
        features, labels = synthetic(n_per_class=30)
        features.iloc[:5] = np.nan
        frame = build_training_frame(features, labels)
        assert frame.excluded["no_measurable_features"] == 5

    def test_partial_measurements_are_kept(self):
        features, labels = synthetic(n_per_class=30, missing_rate=0.0)
        features.iloc[0, 0] = np.nan
        frame = build_training_frame(features, labels)
        assert frame.excluded["no_measurable_features"] == 0
        assert bool(frame.features.iloc[0].isna().any())


# ------------------------------------------------------------------ refusals


class TestRefusals:
    def test_a_single_class_training_set_is_refused(self):
        features, labels = synthetic(n_per_class=40)
        one_sided = [r.model_copy(update={"label": Label.AUTOMATED}) for r in labels]
        with pytest.raises(InsufficientLabelsError, match="only one class"):
            AutomationEnsemble().fit(features, one_sided)

    def test_too_few_examples_is_refused(self):
        features, labels = synthetic(n_per_class=10)
        with pytest.raises(InsufficientLabelsError, match="min_samples_per_class"):
            AutomationEnsemble().fit(features, labels)

    def test_the_refusal_can_be_overridden_deliberately(self):
        features, labels = synthetic(n_per_class=30)
        model = AutomationEnsemble(
            EnsembleConfig(min_samples_per_class=10, n_folds=3, max_iter=25)
        ).fit(features, labels)
        assert model.card is not None

    def test_there_is_no_predict_method(self):
        # Choosing a threshold is a policy decision, not a library default.
        assert not hasattr(AutomationEnsemble, "predict")

    def test_predicting_before_fitting_is_an_error(self):
        features, _ = synthetic(n_per_class=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            AutomationEnsemble().predict_proba(features)


# ------------------------------------------------------------------ training


class TestFitting:
    def test_the_planted_signal_is_found(self, trained):
        model, _, _ = trained
        assert model.card is not None
        assert model.card.evaluation.roc_auc is not None
        assert model.card.evaluation.roc_auc > 0.85

    def test_missing_values_are_learned_from_not_imputed(self, trained):
        # The matrix is 15% holes and nothing filled them in; the model still
        # separates the classes.
        model, features, _ = trained
        assert bool(features.isna().to_numpy().any())
        assert model.card is not None
        assert model.card.n_train == 240

    def test_the_card_records_how_it_was_trained(self, trained):
        model, _, _ = trained
        card = model.card
        assert isinstance(card, ModelCard)
        exported = card.as_dict()
        assert exported["datasets"] == ["synthetic/planted"]
        assert exported["label_methods"] == ["human_annotation"]
        assert exported["features"] == list(FEATURES)
        assert cast("dict[str, Any]", exported["config"])["calibration"] == "isotonic"

    def test_evaluation_is_per_class_and_never_accuracy(self, trained):
        model, _, _ = trained
        assert model.card is not None
        exported = model.card.evaluation.as_dict()
        per_class = cast("dict[str, dict[str, float]]", exported["per_class"])
        assert "accuracy" not in exported
        assert set(per_class) == {"negative", "positive"}
        assert per_class["positive"]["recall"] > 0.6
        assert exported["base_rate"] == 0.5

    def test_the_cross_validation_caveat_is_always_present(self, trained):
        # It is a property of the evaluation design, not of a particular run.
        model, _, _ = trained
        assert model.card is not None
        assert any("splits accounts, not campaigns" in w for w in model.card.warnings)

    def test_imbalance_is_warned_about(self):
        # 28 positives against 120 negatives: enough to train on, lopsided
        # enough that accuracy would be a measure of the majority class.
        features, labels = synthetic(n_per_class=120)
        organic = [r for r in labels if r.label is Label.ORGANIC]
        automated = [r for r in labels if r.label is Label.AUTOMATED][:28]
        keep = [*organic, *automated]
        model = AutomationEnsemble(EnsembleConfig(max_iter=40, n_folds=3)).fit(
            features.loc[[r.account_id for r in keep]], keep
        )
        assert model.card is not None
        assert model.card.class_counts == {0: 120, 1: 28}
        assert any("minority class" in w for w in model.card.warnings)


# ----------------------------------------------------------------- inference


class TestProbabilities:
    def test_probabilities_come_with_a_band(self, trained):
        model, features, _ = trained
        predictions = model.predict_proba(features)
        assert list(predictions.columns) == ["probability", "low", "high", "spread"]
        assert predictions["probability"].between(0, 1).all()
        assert (predictions["low"] <= predictions["high"]).all()

    def test_the_band_is_the_spread_across_folds(self, trained):
        model, features, _ = trained
        predictions = model.predict_proba(features)
        assert (predictions["spread"] == predictions["high"] - predictions["low"]).all()
        assert predictions["spread"].mean() > 0

    def test_the_planted_classes_separate(self, trained):
        model, features, _ = trained
        predictions = model.predict_proba(features)
        bots = predictions.loc[[i for i in predictions.index if i.startswith("bot")]]
        humans = predictions.loc[[i for i in predictions.index if i.startswith("human")]]
        assert bots["probability"].median() > humans["probability"].median() + 0.3

    def test_a_missing_feature_at_inference_is_an_error(self, trained):
        # Quietly filling it with NaN would let a model built on four signals
        # answer from three without saying so.
        model, features, _ = trained
        with pytest.raises(ValueError, match="features missing at inference"):
            model.predict_proba(features.drop(columns=["coord_partner_count"]))

    def test_column_order_does_not_matter(self, trained):
        model, features, _ = trained
        shuffled = features[list(reversed(FEATURES))]
        assert np.allclose(
            model.predict_proba(shuffled)["probability"].to_numpy(),
            model.predict_proba(features)["probability"].to_numpy(),
        )


class TestCalibration:
    def test_perfect_predictions_have_no_calibration_error(self):
        truth = np.array([0, 0, 1, 1])
        assert expected_calibration_error(truth, np.array([0.0, 0.0, 1.0, 1.0])) < 0.01

    def test_confident_and_wrong_is_caught(self):
        truth = np.array([0, 0, 0, 0])
        assert expected_calibration_error(truth, np.array([0.95, 0.95, 0.95, 0.95])) > 0.9

    def test_empty_input_is_zero_not_an_error(self):
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_the_report_carries_a_reliability_curve(self, trained):
        model, _, _ = trained
        assert model.card is not None
        calibration = model.card.evaluation.calibration
        assert calibration.reliability
        assert 0.0 <= calibration.brier <= 1.0
        assert all(n > 0 for _, _, n in calibration.reliability)

    def test_platt_scaling_is_available_for_small_samples(self):
        features, labels = synthetic(n_per_class=40)
        model = AutomationEnsemble(
            EnsembleConfig(calibration="sigmoid", n_folds=3, max_iter=40)
        ).fit(features, labels)
        assert model.card is not None
        assert model.card.config["calibration"] == "sigmoid"


# ------------------------------------------------------------ explainability


class TestExplanations:
    def test_shap_values_line_up_with_the_features(self, trained):
        model, features, _ = trained
        values = model.explain(features.head(20))
        assert values.shape == (20, len(FEATURES))
        assert list(values.columns) == list(FEATURES)

    def test_global_importance_finds_the_planted_signal(self, trained):
        model, features, _ = trained
        ranking = model.global_importance(features)
        names = [name for name, _ in ranking]
        # The fourth feature is pure noise and must not lead.
        assert names[0] != "acct_handle_entropy"
        assert names.index("acct_handle_entropy") >= 2

    def test_explanations_are_returned_not_published(self, trained):
        # Per-account attributions stay with the analyst, like the feature
        # matrix does; the card holds nothing per account.
        model, features, _ = trained
        assert model.card is not None
        exported = str(model.card.as_dict())
        assert "bot0000" not in exported
        assert model.explain(features.head(3)).index.tolist() == features.head(3).index.tolist()


# --------------------------------------------------------------- aggregation


class TestSummary:
    def test_a_corpus_level_view_without_per_account_values(self, trained):
        model, features, _ = trained
        summary = summarise_probabilities(model.predict_proba(features))
        assert summary["n"] == len(features)
        assert len(cast("list[object]", summary["histogram"])) == 10
        assert set(cast("dict[str, float]", summary["quantiles"])) == {"p10", "median", "p90"}
        assert "threshold" in str(summary["caveat"])

    def test_an_empty_input_is_not_a_division_by_zero(self):
        empty = pd.DataFrame({"probability": [], "spread": []})
        assert summarise_probabilities(empty)["n"] == 0


class TestTrainCommand:
    def _corpus_and_labels(self, tmp_path: Path) -> tuple[Path, Path]:
        accounts, posts, rows = [], [], []
        for index in range(90):
            automated = index >= 45
            account_id = f"{'bot' if automated else 'person'}{index:03d}"
            accounts.append(
                make_account(
                    account_id,
                    created_days_ago=20 if automated else 1200,
                    followers=5 if automated else 400,
                    following=900 if automated else 250,
                    handle=f"user{index:07d}" if automated else f"nombre_{index}",
                )
            )
            posts.append(make_post(f"p{index}", account_id, text="Un mensaje de prueba"))
            rows.append(f"{account_id}\t{'bot' if automated else 'human'}")

        corpus_file = write_corpus(build_corpus(accounts, posts), tmp_path / "corpus.json")
        annotations = tmp_path / "labels.dat"
        annotations.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return corpus_file, annotations

    def test_it_trains_and_writes_a_model_card(self, tmp_path, capsys):
        corpus_file, annotations = self._corpus_and_labels(tmp_path)
        card = tmp_path / "card.json"
        code = main(
            [
                "train",
                str(corpus_file),
                str(annotations),
                "--dataset",
                "synthetic/cli",
                "--folds",
                "3",
                "--card",
                str(card),
            ]
        )
        assert code == 0
        payload = json.loads(card.read_text(encoding="utf-8"))
        assert payload["n_train"] == 90
        assert payload["datasets"] == ["synthetic/cli"]
        assert payload["evaluation"]["per_class"].keys() == {"negative", "positive"}
        assert "accuracy" not in payload["evaluation"]
        assert payload["warnings"]
        assert "roc_auc" in capsys.readouterr().out

    def test_the_distribution_it_writes_is_aggregate_only(self, tmp_path):
        corpus_file, annotations = self._corpus_and_labels(tmp_path)
        distribution = tmp_path / "dist.json"
        main(
            [
                "train",
                str(corpus_file),
                str(annotations),
                "--dataset",
                "d",
                "--folds",
                "3",
                "--distribution",
                str(distribution),
            ]
        )
        body = distribution.read_text(encoding="utf-8")
        assert "bot045" not in body
        payload = json.loads(body)
        assert payload["n"] == 90
        assert "threshold" in payload["caveat"]


class TestUnmeasurableFeatures:
    def test_an_all_missing_column_is_dropped_and_recorded(self):
        features, labels = synthetic(n_per_class=60)
        features["temp_burst_share"] = np.nan
        model = AutomationEnsemble(EnsembleConfig(n_folds=3, max_iter=40)).fit(features, labels)
        assert model.card is not None
        assert model.card.dropped_features == ("temp_burst_share",)
        assert "temp_burst_share" not in model.card.features
        assert any("unmeasurable for every labelled" in w for w in model.card.warnings)

    def test_a_dropped_feature_is_not_required_at_inference(self):
        features, labels = synthetic(n_per_class=60)
        features["temp_burst_share"] = np.nan
        model = AutomationEnsemble(EnsembleConfig(n_folds=3, max_iter=40)).fit(features, labels)
        assert len(model.predict_proba(features)) == len(features)

    def test_nothing_measurable_at_all_is_refused(self):
        features, labels = synthetic(n_per_class=60)
        features.iloc[:, :] = np.nan
        with pytest.raises(InsufficientLabelsError):
            AutomationEnsemble(EnsembleConfig(n_folds=3)).fit(features, labels)


class TestDegenerateModels:
    def test_a_constant_predictor_is_called_out(self):
        # scikit-learn's default min_samples_leaf of 20 makes a fold of ~30 rows
        # unsplittable: the tree becomes one leaf, calibration maps everything to
        # the base rate, and a balanced set then yields a respectable-looking
        # Brier score of 0.25 for a model that separated nothing.
        features, labels = synthetic(n_per_class=30, missing_rate=0.0)
        model = AutomationEnsemble(
            EnsembleConfig(n_folds=3, max_iter=40, min_samples_leaf=20, min_samples_per_class=10)
        ).fit(features, labels)
        assert model.card is not None
        assert any("constant probability" in w for w in model.card.warnings)
        assert model.card.evaluation.roc_auc == pytest.approx(0.5)

    def test_the_default_leaf_size_separates_the_planted_signal(self):
        features, labels = synthetic(n_per_class=30, missing_rate=0.0)
        model = AutomationEnsemble(
            EnsembleConfig(n_folds=3, max_iter=40, min_samples_per_class=10)
        ).fit(features, labels)
        assert model.card is not None
        assert not any("constant probability" in w for w in model.card.warnings)
        assert model.card.evaluation.roc_auc is not None
        assert model.card.evaluation.roc_auc > 0.8
