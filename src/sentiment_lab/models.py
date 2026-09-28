"""Model factory: identical TF-IDF settings, three distinct classifiers."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(name: str, config: dict) -> Pipeline:
    classifiers = {
        "multinomial_nb": MultinomialNB(),
        "logistic_regression": LogisticRegression(solver="liblinear", max_iter=2000, random_state=config["seed"]),
        "linear_svm": LinearSVC(dual="auto", max_iter=5000, random_state=config["seed"]),
    }
    options = dict(config["features"])
    options["ngram_range"] = tuple(options["ngram_range"])
    # Retain negations (no English stop-word list). Float32 keeps sparse features compact.
    features = TfidfVectorizer(dtype=np.float32, **options)
    return Pipeline([("tfidf", features), ("clf", classifiers[name])])
