"""Readable, consistent PNG/SVG figures; English labels avoid missing fonts."""
import matplotlib

# Agg 是无界面绘图后端，必须在导入 pyplot 前设置，支持命令行和无显示器环境。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import Paths

NAMES = {"multinomial_nb": "Naive Bayes", "logistic_regression": "Logistic regression", "linear_svm": "Linear SVM"}
# NAMES 将内部算法键映射为展示名；COLORS 按 summary['models'] 的记录位置依次使用。
# 所有模型比较子图沿用同一记录顺序，避免颜色与模型标签对应关系发生变化。
COLORS = ["#B64342", "#42949E", "#0F4D92"]


def save_figure(fig, folder, name):
    # 输入 Figure、输出目录和不带扩展名的文件名；调用方负责创建 folder。
    # tight_layout 按轴标题和刻度调整留白，dpi=300 决定 PNG 像素密度，SVG 保留矢量图形。
    # 同时输出高分辨率 PNG 和可缩放 SVG；布局调整后关闭 Figure，释放绘图资源。
    fig.tight_layout(pad=1.5)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def make_figures(paths: Paths, summary: dict) -> None:
    # 输入目录和训练汇总，返回 None；依次输出模型比较、混淆矩阵、ROC、数据概况、词权重。
    # summary 提供聚合指标；逐条预测用于 ROC，清洗正文用于长度直方图，两者不能省略。
    folder = paths.reports / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    # rcParams 控制后续图的统一字体与轴风格；英语图中文字避免运行环境缺中文字体。
    # svg.fonttype='none' 保留文本为 SVG 文字，查看设备仍需具备相应字体。
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 1.2, "legend.frameon": False, "svg.fonttype": "none"})
    records = summary["models"]
    names = [NAMES[row["model"]] for row in records]
    # 模型对比图同时展示测试效果与包含 CV/重拟合的总搜索耗时。
    # Accuracy 与 macro-F1 并列，虚线标记多数类基线，颜色在所有图中保持一致。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    # fig 是整张画布，axes[0]/axes[1] 是左右子图；figsize 的单位是英寸。
    x = np.arange(len(records))
    # x=[0,1,2] 给出各模型的中心位置；两指标用正负偏移并列，hatch 用纹理辅助区分。
    for offset, metric, label, hatch in [(-0.19, "accuracy", "Accuracy", ""), (0.19, "macro_f1", "Macro-F1", "//")]:
        values = [row[metric] for row in records]
        bars = axes[0].bar(x + offset, values, width=0.36, color=COLORS, edgecolor="#272727", hatch=hatch, label=label)
        # bar_label 标注柱高，fmt 控制显示精度，原始指标仍保留在 metrics.json 中。
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 1.04), ylabel="Test score", title="A  |  Held-out performance")
    axes[0].axhline(summary["baseline"]["accuracy"], ls=":", color="gray", label="Majority accuracy")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    # 图例放在坐标区外的下方，避免遮住柱子和数字；纵轴统一从 0 开始便于比较。
    bars = axes[1].bar(x, [row["search_seconds"] for row in records], color=COLORS, edgecolor="#272727")
    axes[1].bar_label(bars, fmt="%.1f s", padding=3)
    axes[1].set(xticks=x, xticklabels=names, ylabel="Seconds", title="B  |  CV search + final fit")
    axes[1].margins(y=0.2)
    save_figure(fig, folder, "model_comparison")
    # 各模型混淆矩阵共用色阶上限，避免相同颜色在不同子图代表不同数量。
    # 横轴预测、纵轴真实；每格同时标注原始计数，便于区分两种误判方向。
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, row in zip(axes, records):
        # zip 将三个子图与三种模型记录一一配对，不按模型分数重新排列。
        matrix = np.asarray(row["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(np.asarray(r["confusion_matrix"]).max() for r in records))
        for i in range(2):
            for j in range(2):
                # i 是真实类别行，j 是预测类别列；imshow 内文字坐标写成 (j,i)。
                # 在较深色格子用白字提高可读性，颜色不改变矩阵统计含义。
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "#272727")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
               yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual", title=NAMES[row["model"]])
    save_figure(fig, folder, "confusion_matrices")
    # ROC 从保存的逐条连续分数重新计算；汇总 AUC 只能作为图例，无法还原整条曲线。
    # 对角虚线是无排序能力的参照，各模型必须使用同一批测试标签。
    predictions = pd.read_csv(paths.results / "test_predictions.csv.gz")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for row, color in zip(records, COLORS):
        # roc_curve 扫描不同分数阈值：FPR=FP/(FP+TN)，TPR=TP/(TP+FN)。
        # 正面是标签 1，越高分数越倾向正面；曲线越靠近左上角，当前排序通常越好。
        fpr, tpr, _ = roc_curve(predictions.label, predictions[f"{row['model']}_score"])
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{NAMES[row['model']]}  AUC={row['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on the independent test set")
    ax.legend(loc="lower right")
    save_figure(fig, folder, "roc_curves")
    # 类别数量来自清洗审计，词数分布来自清洗后的正文；图中仅显示 0–1500 词范围，
    # 这只是绘图范围限制，并没有从训练/测试数据中删除更长评论。
    data = pd.read_csv(paths.processed / "reviews.csv.gz", keep_default_na=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = summary["data_audit"]["counts"]
    for offset, label, color in [(-0.18, "0", "#B64342"), (0.18, "1", "#0F4D92")]:
        bars = axes[0].bar(np.arange(2) + offset, [counts[s][label] for s in ["train", "test"]], width=0.34, color=color,
                           label="Negative" if label == "0" else "Positive")
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(xticks=[0, 1], xticklabels=["Train", "Test"], ylabel="Reviews", title="A  |  Cleaned class balance")
    axes[0].margins(y=0.2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    for split, color in [("train", "#42949E"), ("test", "#0F4D92")]:
        lengths = data.loc[data.split.eq(split)].text.str.split().str.len()
        # bin 边界每 50 词一个间隔，两划分使用相同边界；alpha=0.5 让重叠柱形仍可看见。
        axes[1].hist(lengths, bins=np.arange(0, 1501, 50), alpha=0.5, color=color, label=split)
    axes[1].set(xlabel="Words per review (0–1500 displayed)", ylabel="Reviews", title="B  |  Review length")
    axes[1].legend()
    save_figure(fig, folder, "data_overview")
    # 取所选模型每类最靠前的 12 个特征并反转显示顺序，将最强指标放在图的上方。
    # 系数/对数概率差体现训练数据中的关联，不能解释为词语的因果作用。
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, polarity, color in zip(axes, ["negative", "positive"], ["#B64342", "#0F4D92"]):
        # 数据已由 explain_features 按极端权重排序，[:12] 取前 12，[::-1] 仅反转显示顺序。
        features = summary["feature_weights"][polarity][:12][::-1]
        ax.barh([f["term"] for f in features], [f["weight"] for f in features], color=color)
        ax.set(xlabel="Coefficient / log-probability difference", title=f"{polarity.title()} indicators")
    save_figure(fig, folder, "feature_weights")
