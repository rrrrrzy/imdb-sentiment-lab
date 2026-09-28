"""Readable, consistent PNG/SVG figures; English labels avoid missing fonts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import Paths

NAMES = {"multinomial_nb": "Naive Bayes", "logistic_regression": "Logistic regression", "linear_svm": "Linear SVM"}
COLORS = ["#B64342", "#42949E", "#0F4D92"]


def save_figure(fig, folder, name):
    fig.tight_layout(pad=1.5)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def make_figures(paths: Paths, summary: dict) -> None:
    folder = paths.reports / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 1.2, "legend.frameon": False, "svg.fonttype": "none"})
    records = summary["models"]
    names = [NAMES[row["model"]] for row in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(records))
    for offset, metric, label, hatch in [(-0.19, "accuracy", "Accuracy", ""), (0.19, "macro_f1", "Macro-F1", "//")]:
        values = [row[metric] for row in records]
        bars = axes[0].bar(x + offset, values, width=0.36, color=COLORS, edgecolor="#272727", hatch=hatch, label=label)
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 1.04), ylabel="Test score", title="A  |  Held-out performance")
    axes[0].axhline(summary["baseline"]["accuracy"], ls=":", color="gray", label="Majority accuracy")
    axes[0].legend(loc="lower right", fontsize=9)
    bars = axes[1].bar(x, [row["search_seconds"] for row in records], color=COLORS, edgecolor="#272727")
    axes[1].bar_label(bars, fmt="%.1f s", padding=3)
    axes[1].set(xticks=x, xticklabels=names, ylabel="Seconds", title="B  |  CV search + final fit")
    axes[1].margins(y=0.2)
    save_figure(fig, folder, "model_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, row in zip(axes, records):
        matrix = np.asarray(row["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(np.asarray(r["confusion_matrix"]).max() for r in records))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "#272727")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
               yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual", title=NAMES[row["model"]])
    save_figure(fig, folder, "confusion_matrices")
    predictions = pd.read_csv(paths.results / "test_predictions.csv.gz")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for row, color in zip(records, COLORS):
        fpr, tpr, _ = roc_curve(predictions.label, predictions[f"{row['model']}_score"])
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{NAMES[row['model']]}  AUC={row['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on the independent test set")
    ax.legend(loc="lower right")
    save_figure(fig, folder, "roc_curves")
    data = pd.read_csv(paths.processed / "reviews.csv.gz", keep_default_na=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = summary["data_audit"]["counts"]
    for offset, label, color in [(-0.18, "0", "#B64342"), (0.18, "1", "#0F4D92")]:
        bars = axes[0].bar(np.arange(2) + offset, [counts[s][label] for s in ["train", "test"]], width=0.34, color=color,
                           label="Negative" if label == "0" else "Positive")
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(xticks=[0, 1], xticklabels=["Train", "Test"], ylabel="Reviews", title="A  |  Cleaned class balance")
    axes[0].margins(y=0.2)
    axes[0].legend(loc="lower right")
    for split, color in [("train", "#42949E"), ("test", "#0F4D92")]:
        lengths = data.loc[data.split.eq(split)].text.str.split().str.len()
        axes[1].hist(lengths, bins=np.arange(0, 1501, 50), alpha=0.5, color=color, label=split)
    axes[1].set(xlabel="Words per review (0–1500 displayed)", ylabel="Reviews", title="B  |  Review length")
    axes[1].legend()
    save_figure(fig, folder, "data_overview")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, polarity, color in zip(axes, ["negative", "positive"], ["#B64342", "#0F4D92"]):
        features = summary["feature_weights"][polarity][:12][::-1]
        ax.barh([f["term"] for f in features], [f["weight"] for f in features], color=color)
        ax.set(xlabel="Coefficient / log-probability difference", title=f"{polarity.title()} indicators")
    save_figure(fig, folder, "feature_weights")
