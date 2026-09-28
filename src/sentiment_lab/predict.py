"""Inference uses the persisted pipeline and exactly the same text cleanup."""
import joblib

from .config import Paths
from .data.prepare import clean_text
from .evaluation import continuous_scores
from .io import read_json


def load_selected_model(paths: Paths):
    selection = read_json(paths.results / "selection.json")
    name = selection["winner"]
    # joblib/pickle must only be loaded from this project's trusted training output.
    return name, joblib.load(paths.models / f"{name}.joblib")


def predict_review(name: str, model, text: str) -> dict:
    if not isinstance(text, str) or not 2 <= len(text) <= 10000:
        raise ValueError("Please enter 2–10000 characters of English review text")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("No review text remains after HTML cleaning")
    features = model.named_steps["tfidf"].transform([cleaned])
    if features.nnz == 0:
        raise ValueError("No known English features found; please use an English movie review")
    return {"model": name, "label": int(model.predict([cleaned])[0]),
            "score": float(continuous_scores(model, [cleaned])[0]),
            "score_type": "decision score" if hasattr(model, "decision_function") else "positive probability"}
