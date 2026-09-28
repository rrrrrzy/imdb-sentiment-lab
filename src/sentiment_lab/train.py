"""Tune only on training folds, freeze selection, then evaluate the holdout."""
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from .config import Paths
from .evaluation import bootstrap_accuracy, continuous_scores, metrics
from .io import read_json, write_json
from .models import build_pipeline

LOGGER = logging.getLogger(__name__)


def explain_features(model, count: int = 18) -> dict:
    classifier = model.named_steps["clf"]
    weights = (classifier.coef_[0] if hasattr(classifier, "coef_")
               else classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0])
    terms = model.named_steps["tfidf"].get_feature_names_out()
    order = np.argsort(weights)
    return {"negative": [{"term": str(terms[i]), "weight": float(weights[i])} for i in order[:count]],
            "positive": [{"term": str(terms[i]), "weight": float(weights[i])} for i in order[-count:][::-1]]}


def run_experiment(paths: Paths, config: dict) -> dict:
    data = pd.read_csv(paths.processed / "reviews.csv.gz", keep_default_na=False)
    train = data.loc[data.split.eq("train")]
    test = data.loc[data.split.eq("test")]
    if len(data) < config["min_samples"] or train.empty or test.empty:
        raise ValueError("Insufficient prepared data; run prepare first")
    if set(train.text_hash) & set(test.text_hash):
        raise ValueError("Training/test overlap detected")
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"])
    tuned, records = {}, []
    # Convergence failures must surface as errors, not produce a misleading final report.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        for name, grid in config["models"].items():
            LOGGER.info("Tuning %s: %s folds, %s candidates", name, config["cv_folds"], len(next(iter(grid.values()))))
            started = time.perf_counter()
            search = GridSearchCV(build_pipeline(name, config), grid, scoring="f1_macro", cv=cv,
                                  n_jobs=config["n_jobs"], refit=True, error_score="raise", return_train_score=True)
            search.fit(train.text, train.label)
            elapsed = time.perf_counter() - started
            tuned[name] = search.best_estimator_
            results = pd.DataFrame(search.cv_results_)
            results.to_csv(paths.results / f"cv_{name}.csv", index=False)
            record = {"model": name, "best_params": search.best_params_,
                      "cv_macro_f1": float(search.best_score_),
                      "cv_std": float(results.loc[search.best_index_, "std_test_score"]),
                      "cv_train_macro_f1": float(results.loc[search.best_index_, "mean_train_score"]),
                      "search_seconds": elapsed, "refit_seconds": float(search.refit_time_),
                      "vocabulary_size": len(search.best_estimator_.named_steps["tfidf"].vocabulary_)}
            records.append(record)
            joblib.dump(search.best_estimator_, paths.models / f"{name}.joblib", compress=3)
            LOGGER.info("%s CV macro-F1 %.4f; search %.1fs", name, search.best_score_, elapsed)
    winner = max(records, key=lambda row: row["cv_macro_f1"])["model"]
    # Persist the decision before reading any test predictions.
    write_json(paths.results / "selection.json", {"winner": winner, "criterion": f"training {config['cv_folds']}-fold CV macro-F1",
                                                 "test_used_for_selection": False,
                                                 "cv_folds": config["cv_folds"]})
    predictions = {}
    prediction_frame = test[["id", "label"]].reset_index(drop=True).copy()
    for row in records:
        name = row["model"]
        started = time.perf_counter()
        pred = tuned[name].predict(test.text)
        scores = continuous_scores(tuned[name], test.text)
        row["predict_and_score_seconds"] = time.perf_counter() - started
        row.update(metrics(test.label, pred, scores))
        predictions[name] = pred
        prediction_frame[f"{name}_prediction"] = pred
        prediction_frame[f"{name}_score"] = scores
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train.label)
    baseline_prediction = baseline.predict(np.zeros((len(test), 1)))
    baseline_result = metrics(test.label, baseline_prediction, np.full(len(test), 0.5))
    baseline_result["model"] = "majority_baseline"
    predictions["majority_baseline"] = baseline_prediction
    uncertainty = bootstrap_accuracy(test.label, predictions, winner, config["bootstrap_repeats"], config["seed"])
    pred = predictions[winner]
    errors = test.loc[pred != test.label.to_numpy(), ["id", "label", "text"]].copy()
    errors["prediction"] = pred[pred != test.label.to_numpy()]
    errors["text"] = errors.text.str.slice(0, 1500)
    errors.head(80).to_csv(paths.results / "error_examples.csv", index=False)
    lengths = test.text.str.split().str.len().to_numpy()
    length_analysis = []
    for label, lower, upper in [("<100 words", 0, 100), ("100–299 words", 100, 300), (">=300 words", 300, np.inf)]:
        mask = (lengths >= lower) & (lengths < upper)
        length_analysis.append({"group": label, "n": int(mask.sum()),
                                "accuracy": float((pred[mask] == test.label.to_numpy()[mask]).mean()) if mask.any() else None})
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
               "winner": winner, "train_rows": len(train), "test_rows": len(test),
               "models": records, "baseline": baseline_result, "uncertainty": uncertainty,
               "length_analysis": length_analysis, "feature_weights": explain_features(tuned[winner]),
               "data_audit": read_json(paths.results / "data_audit.json"),
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "packages": {name: version(name) for name in ["scikit-learn", "numpy", "pandas", "matplotlib", "joblib"]}}}
    write_json(paths.results / "metrics.json", summary)
    write_json(paths.results / "run_config.json", config)
    prediction_frame.to_csv(paths.results / "test_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    metric_columns = ["model", "cv_macro_f1", "cv_std", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc",
                      "search_seconds", "refit_seconds", "predict_and_score_seconds", "vocabulary_size"]
    pd.DataFrame(records)[metric_columns].to_csv(paths.results / "model_comparison.csv", index=False)
    return summary
