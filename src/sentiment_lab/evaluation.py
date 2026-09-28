"""Holdout metrics and paired uncertainty estimates, independent of training."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def continuous_scores(model, texts) -> np.ndarray:
    # 输入为已拟合模型与 N 条文本，返回长度 N 的分数数组，保持原始文本顺序。
    # hasattr 检查 Pipeline 最终分类器是否提供决策函数；本项目模型均为二分类。
    # 项目标签固定为 0/1，连续分数越大越倾向正面，供 ROC 排序和演示使用。
    # 优先取决策函数：LR 和 SVM 均走此分支，分数不表示概率；NB 则取正类概率列。
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    # predict_proba 返回 (N,2) 矩阵；[:,1] 取所有行的正类列，前提是类别顺序 [0,1]。
    return model.predict_proba(texts)[:, 1]


def metrics(y_true, predicted, scores) -> dict:
    # 三个输入都是等长且逐行对应的序列：真实标签、硬预测标签、正面倾向连续分数。
    # 返回 JSON 可序列化的字典；这里不拟合模型、不选择阈值，也不修改输入。
    # TP=真实正面且预测正面，FP=真实负面却预测正面，FN=真实正面却预测负面。
    # Accuracy 看整体正确比例；Precision/Recall/F1 默认以标签 1（正面）为正类。
    # Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1 为二者调和平均；
    # macro-F1 对正负两类 F1 等权平均，避免只关注其中一类。
    # zero_division=0 规定无正面预测等情况下的结果，防止分母为零产生未定义指标。
    # 例：真实 [0,0,1,1]、预测 [0,1,1,1] → Accuracy=3/4，Precision=2/3，
    # Recall=2/2，混淆矩阵 [[1,1],[0,2]]。这些是公式示例，不是实际实验指标。
    return {"accuracy": float(accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
            # AUC 衡量连续分数的排序能力，不能把 0/1 硬预测当作完整排序分数。
            "roc_auc": float(roc_auc_score(y_true, scores)),
            # 行=真实标签，列=预测标签；顺序 [0,1] 得到 [[TN,FP],[FN,TP]]。
            "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y_true, predicted, labels=[0, 1],
                target_names=["negative", "positive"], output_dict=True, zero_division=0)}


def bootstrap_accuracy(y_true, predictions: dict, winner: str, repeats: int, seed: int) -> dict:
    """Same resampled review indices for all models: paired percentile intervals.

    Review-level bootstrap assumes independent reviews; it does not measure
    training randomness, film clusters, or domain shift.
    """
    # 输入 predictions 为 {模型名: 长度 N 的预测序列}，winner 是其中的已选模型键。
    # 输出 accuracy_ci[模型名]=[下界,上界]，以及 winner_minus_other_accuracy_ci。
    # 所有区间都用 0–1 的准确率单位；报告乘 100 后，差值单位才是“百分点”。
    # 不重新训练模型，只估计这批测试评论的抽样不确定性；假设评论相互独立，
    # 不涵盖训练随机性、同电影聚类或跨领域变化。固定 seed 使重采样可复现。
    truth = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    # 先把预测转成逐条“是否正确”的布尔数组，后续均值就是对应样本的准确率。
    correct = {name: np.asarray(pred) == truth for name, pred in predictions.items()}
    # samples[模型名] 长度为 repeats，依次存每轮准确率；np.empty 先分配空间，
    # 后续每个位置都会在循环中赋值，初始内容不能拿来计算。
    samples = {name: np.empty(repeats) for name in predictions}
    for iteration in range(repeats):
        # 有放回抽取与原测试集等长的一组索引，同一评论可出现多次或完全不出现。
        # 本轮所有模型共享索引，确保差值比较基于同一组评论，而非不同抽样难度。
        indices = rng.integers(0, len(truth), len(truth))
        # integers 的上界不包含 len(truth)，所以所有索引落在 [0,N-1]。
        # 例如 N=4，本轮可能抽到 [2,2,0,3]；correct[name][indices] 按此顺序重复取样。
        for name in predictions:
            samples[name][iteration] = correct[name][indices].mean()
    # 布尔均值可直接计分：例如 [True,True,False,True].mean() = 3/4。
    # 2.5% 与 97.5% 分位数组成 95% 百分位区间。
    # 模型差值逐轮计算 winner-other 后再取分位数，不能直接相减两个独立区间端点；
    # 差值区间包含 0 时，当前重采样结果不足以明确区分两者。
    return {"repetitions": repeats, "confidence_level": 0.95,
            "accuracy_ci": {name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in samples.items()},
            "winner_minus_other_accuracy_ci": {
                name: np.quantile(samples[winner] - values, [0.025, 0.975]).tolist()
                for name, values in samples.items() if name != winner}}
