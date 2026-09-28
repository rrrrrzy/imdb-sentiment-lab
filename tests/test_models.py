import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sentiment_lab.config import load_config
from sentiment_lab.evaluation import bootstrap_accuracy, continuous_scores, metrics
from sentiment_lab.models import build_pipeline
from sentiment_lab.predict import predict_review
from pathlib import Path

CONFIG = load_config(Path(__file__).parents[1] / "configs" / "experiment.json")
TEXTS = ["wonderful happy good acting", "bad awful boring acting", "good wonderful interesting movie", "awful bad dull movie"] * 6
LABELS = np.array([1, 0, 1, 0] * 6)


@pytest.mark.parametrize("name", list(CONFIG["models"]))
def test_pipeline_cv_and_inference_exclude_unseen_test_vocabulary(name):
    model = build_pipeline(name, CONFIG)
    search = GridSearchCV(model, CONFIG["models"][name], cv=StratifiedKFold(3, shuffle=True, random_state=42),
                          scoring="f1_macro", error_score="raise")
    search.fit(TEXTS, LABELS)
    fitted = search.best_estimator_
    predictions = fitted.predict(["wonderful good unseenholdoutword", "bad awful"])
    assert predictions.tolist() == [1, 0]
    assert "unseenholdoutword" not in fitted.named_steps["tfidf"].vocabulary_
    scores = continuous_scores(fitted, ["wonderful good", "bad awful"])
    assert scores[0] > scores[1]
    with pytest.raises(ValueError, match="No known English"):
        predict_review(name, fitted, "未知词语")


def test_metric_orientation_and_positive_class():
    result = metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.8, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2/3)
    assert result["roc_auc"] == 1.0


def test_paired_bootstrap_identical_models_have_zero_difference():
    predictions = {"first": [0, 1, 1, 1], "second": [0, 1, 1, 1]}
    result = bootstrap_accuracy([0, 0, 1, 1], predictions, "first", 100, 42)
    assert result["winner_minus_other_accuracy_ci"]["second"] == [0.0, 0.0]
    assert result == bootstrap_accuracy([0, 0, 1, 1], predictions, "first", 100, 42)
