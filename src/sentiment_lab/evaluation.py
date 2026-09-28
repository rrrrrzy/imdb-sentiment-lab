"""Holdout metrics and paired uncertainty estimates, independent of training."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def continuous_scores(model, texts) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    return model.predict_proba(texts)[:, 1]


def metrics(y_true, predicted, scores) -> dict:
    return {"accuracy": float(accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y_true, predicted, labels=[0, 1],
                target_names=["negative", "positive"], output_dict=True, zero_division=0)}


def bootstrap_accuracy(y_true, predictions: dict, winner: str, repeats: int, seed: int) -> dict:
    """Same resampled review indices for all models: paired percentile intervals.

    Review-level bootstrap assumes independent reviews; it does not measure
    training randomness, film clusters, or domain shift.
    """
    truth = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    correct = {name: np.asarray(pred) == truth for name, pred in predictions.items()}
    samples = {name: np.empty(repeats) for name in predictions}
    for iteration in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        for name in predictions:
            samples[name][iteration] = correct[name][indices].mean()
    return {"repetitions": repeats, "confidence_level": 0.95,
            "accuracy_ci": {name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in samples.items()},
            "winner_minus_other_accuracy_ci": {
                name: np.quantile(samples[winner] - values, [0.025, 0.975]).tolist()
                for name, values in samples.items() if name != winner}}
