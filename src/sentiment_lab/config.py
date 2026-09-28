"""Explicit configuration and project paths."""
from dataclasses import dataclass
from pathlib import Path

from .io import read_json


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def results(self) -> Path:
        return self.root / "artifacts" / "results"

    @property
    def models(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def load_config(path: Path) -> dict:
    config = read_json(path)
    if config["cv_folds"] < 2 or config["min_samples"] < 10000:
        raise ValueError("cv_folds must be >= 2 and min_samples must be >= 10000")
    if set(config["models"]) != {"multinomial_nb", "logistic_regression", "linear_svm"}:
        raise ValueError("The experiment requires all three supported models")
    if config["bootstrap_repeats"] < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    return config
