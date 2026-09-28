"""Clean HTML and whitespace while retaining negations; audit duplicates."""
import hashlib
import html
import re

import pandas as pd

from ..config import Paths
from ..io import write_json


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]*>", " ", value)
    value = html.unescape(value).lower()
    return re.sub(r"\s+", " ", value).strip()


def audit_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if raw["id"].duplicated().any() or not set(raw["label"].unique()) <= {0, 1}:
        raise ValueError("IDs must be unique and labels must be binary")
    frame = raw.copy()
    frame["text"] = frame["text"].map(clean_text)
    audit = {"original_rows": len(frame), "empty_removed": int(frame.text.eq("").sum()),
             "within_split": {}}
    frame = frame.loc[frame.text.ne("")].copy()
    frame["text_hash"] = frame.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    parts = {}
    for split in ("train", "test"):
        subset = frame.loc[frame.split.eq(split)].copy()
        conflicts = subset.groupby("text_hash").label.nunique()
        conflicts = set(conflicts[conflicts > 1].index)
        ambiguous = subset.text_hash.isin(conflicts)
        audit["within_split"][split] = {"conflicting_rows_removed": int(ambiguous.sum())}
        subset = subset.loc[~ambiguous]
        audit["within_split"][split]["duplicate_rows_removed"] = int(subset.text_hash.duplicated().sum())
        parts[split] = subset.drop_duplicates("text_hash")
    # Honor the original test boundary; never move test reviews into training.
    overlap = parts["test"].text_hash.isin(set(parts["train"].text_hash))
    audit["cross_split_test_duplicates_removed"] = int(overlap.sum())
    parts["test"] = parts["test"].loc[~overlap]
    clean = pd.concat(parts.values(), ignore_index=True)
    audit["clean_rows"] = len(clean)
    audit["counts"] = {split: {str(label): int(count) for label, count in part.label.value_counts().sort_index().items()}
                       for split, part in parts.items()}
    audit["word_length"] = {
        split: {"median": float(part.text.str.split().str.len().median()),
                "mean": float(part.text.str.split().str.len().mean()),
                "p95": float(part.text.str.split().str.len().quantile(0.95))}
        for split, part in parts.items()}
    if set(parts["train"].text_hash) & set(parts["test"].text_hash):
        raise AssertionError("Text leakage across train/test")
    return clean, audit


def prepare(raw: pd.DataFrame, paths: Paths, min_samples: int) -> pd.DataFrame:
    clean, audit = audit_and_clean(raw)
    if len(clean) < min_samples:
        raise ValueError(f"At least {min_samples} clean reviews are required")
    paths.processed.mkdir(parents=True, exist_ok=True)
    clean.to_csv(paths.processed / "reviews.csv.gz", index=False, compression="gzip")
    write_json(paths.results / "data_audit.json", audit)
    return clean
