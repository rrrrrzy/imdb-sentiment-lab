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
    # 输入已拟合 Pipeline 和每个方向最多显示的特征数 count。
    # 输出 {'negative': [{'term':词组,'weight':权重}, ...], 'positive': [...]}，
    # 与后续网页特征框及横向条形图的输入结构一致。
    classifier = model.named_steps["clf"]
    # 二分类 LR/SVM 的 coef_[0] 与正类 1 对应，权重越大越偏向正面。
    # NB 没有 coef_，用 log P(词|正面) - log P(词|负面) 表示词语的相对倾向。
    weights = (classifier.coef_[0] if hasattr(classifier, "coef_")
               else classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0])
    # LR/SVM 的决策函数可写为 s(x)=w·x+b，weights 即长度 V 的 w；这里不展示截距 b。
    # NB 的差值来自两个类别的特征对数概率，不能与线性模型的系数绝对值直接比较。
    # 特征名顺序与权重列一一对应；从排序两端取出权重最低和最高的词/词组。
    # 这些关联用于定性解释，不是因果证据，也不参与模型选择。
    terms = model.named_steps["tfidf"].get_feature_names_out()
    order = np.argsort(weights)
    # argsort 返回排序后的原列索引，而不是权重本身；terms[i] 才能与 weights[i] 配对。
    # order[:count] 从最低权重开始，order[-count:][::-1] 从最高权重开始。
    # negative/positive 表示排序方向；代码没有额外检查每个取出权重的正负号。
    return {"negative": [{"term": str(terms[i]), "weight": float(weights[i])} for i in order[:count]],
            "positive": [{"term": str(terms[i]), "weight": float(weights[i])} for i in order[-count:][::-1]]}


def run_experiment(paths: Paths, config: dict) -> dict:
    # 输入：Paths 项目目录对象和完整实验配置；依赖 prepare 已生成清洗 CSV 与审计文件。
    # 返回 summary，同时写入模型、CV 表、选择记录、测试指标及预测等可复核产物。
    # 执行顺序：读数据 → 训练 CV/重拟合 → 冻结选择 → 测试评估 → 误判/区间 → 保存。
    # 此函数不收集网络数据，CLI 的 run 命令才负责在训练前串联采集和清洗。
    # 保留空字符串等原始文本值，按已有 split 取训练/测试数据，不重新随机切分。
    data = pd.read_csv(paths.processed / "reviews.csv.gz", keep_default_na=False)
    # split.eq('train') 等价于逐行比较 split == 'train'；loc 用掩码筛选整行。
    # keep_default_na=False 还防止评论正文中的 'NA' 等字符串被 pandas 当作缺失值。
    train = data.loc[data.split.eq("train")]
    test = data.loc[data.split.eq("test")]
    if len(data) < config["min_samples"] or train.empty or test.empty:
        raise ValueError("Insufficient prepared data; run prepare first")
    # 即使用户直接提供 processed 文件，也必须在拟合前检查两划分的正文哈希交集。
    if set(train.text_hash) & set(test.text_hash):
        raise ValueError("Training/test overlap detected")
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    # 分层保持各折类别比例；固定 seed 后，三个算法使用可复现的相同划分规则。
    # cv 只传入 train 数据，独立 test 集不参与搜索。
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"])
    # tuned={算法名: 已重拟合的最佳 Pipeline}，用于后面的统一测试推理；
    # records=[每种算法的汇总字典]，先存 CV/耗时字段，测试结束后再补充测试指标。
    tuned, records = {}, []
    # Convergence failures must surface as errors, not produce a misleading final report.
    # 限制底层 BLAS 线程数，减少笔记本资源争用；收敛警告提升为异常，避免把
    # 尚未正常收敛的模型当成成功结果。GridSearchCV 的拟合错误也直接抛出。
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        for name, grid in config["models"].items():
            LOGGER.info("Tuning %s: %s folds, %s candidates", name, config["cv_folds"], len(next(iter(grid.values()))))
            started = time.perf_counter()
            # 按验证折 macro-F1 比较参数；refit=True 会用最佳参数在完整训练集上
            # 再拟合一次 Pipeline。return_train_score 保留训练/验证差距供报告分析。
            # grid 示例 {'clf__C':[0.5,2.0]}：每个 C 在每折评估一次。
            # 默认每种算法 2 候选×3 折=6 次 CV 拟合，再做 1 次完整训练集重拟合；
            # 三算法合计 18 次 CV 拟合和 3 次重拟合，每次都包含重新学习 TF-IDF。
            # 仅传 train.text 和 train.label，rating/id 等列即使在表中也不会进入特征。
            search = GridSearchCV(build_pipeline(name, config), grid, scoring="f1_macro", cv=cv,
                                  n_jobs=config["n_jobs"], refit=True, error_score="raise", return_train_score=True)
            search.fit(train.text, train.label)
            # 此耗时包含每个候选每折的向量化、分类器拟合及最终重拟合，
            # 不能将其解释为单次分类器训练耗时；单独的重拟合耗时见 refit_seconds。
            elapsed = time.perf_counter() - started
            tuned[name] = search.best_estimator_
            # best_estimator_ 是完整训练集重拟合后的对象，不是某个验证折留下的模型。
            # cv_results_ 的每行对应一个候选，含 params、split0_test_score 等逐折字段；
            # best_index_ 指向该算法验证均值最高的候选行，不表示评论行号。
            results = pd.DataFrame(search.cv_results_)
            # 保留完整候选与逐折结果，便于独立核对最佳均值；cv_std 是折间标准差，
            # 不等同于后续测试准确率的 bootstrap 置信区间。
            results.to_csv(paths.results / f"cv_{name}.csv", index=False)
            record = {"model": name, "best_params": search.best_params_,
                      "cv_macro_f1": float(search.best_score_),
                      "cv_std": float(results.loc[search.best_index_, "std_test_score"]),
                      "cv_train_macro_f1": float(results.loc[search.best_index_, "mean_train_score"]),
                      "search_seconds": elapsed, "refit_seconds": float(search.refit_time_),
                      "vocabulary_size": len(search.best_estimator_.named_steps["tfidf"].vocabulary_)}
            # best_score_ 是最佳候选各验证折的平均 macro-F1，cv_train_macro_f1 则是
            # 同一候选各训练折的平均值。最终 vocabulary_size 来自重拟合后的词表，
            # 与各 CV 折的词表大小不必相等，也可能小于 max_features 上限。
            records.append(record)
            # 保存整个 Pipeline（含词表、IDF 和分类器），推理时无需重新拟合特征。
            joblib.dump(search.best_estimator_, paths.models / f"{name}.joblib", compress=3)
            LOGGER.info("%s CV macro-F1 %.4f; search %.1fs", name, search.best_score_, elapsed)
    winner = max(records, key=lambda row: row["cv_macro_f1"])["model"]
    # max 比较所有算法的最佳 CV 均值；若完全并列，会保留 records 中最先遇到的模型，
    # 不用测试集作为破平规则。后续的 winner 在本次流程中保持不变。
    # Persist the decision before reading any test predictions.
    # 到此只按训练 CV 选模型并落盘。即使其他模型稍后的测试分数更高，也不改选。
    write_json(paths.results / "selection.json", {"winner": winner, "criterion": f"training {config['cv_folds']}-fold CV macro-F1",
                                                 "test_used_for_selection": False,
                                                 "cv_folds": config["cv_folds"]})
    predictions = {}
    # 所有模型共用同一测试行顺序，保存 ID、真实标签、硬预测与连续分数，
    # 后续可以不加载模型就复算指标、ROC 和模型间的配对差值。
    prediction_frame = test[["id", "label"]].reset_index(drop=True).copy()
    # 原表索引在筛选 test 后可能不从 0 开始；reset_index 只重置输出表的行号。
    # predict 返回 NumPy 数组，赋入 DataFrame 时按位置放置，因此与 test 的当前顺序一致。
    for row in records:
        name = row["model"]
        started = time.perf_counter()
        # 硬标签用于准确率/F1/混淆矩阵；连续分数用于 ROC-AUC，不能相互替代。
        pred = tuned[name].predict(test.text)
        scores = continuous_scores(tuned[name], test.text)
        # 该耗时同时包含硬预测和连续打分，两次调用可能分别执行 TF-IDF transform；
        # 不能解释为一次单条评论的预测延迟。
        row["predict_and_score_seconds"] = time.perf_counter() - started
        row.update(metrics(test.label, pred, scores))
        # row 是 records 中字典的引用，update 会把测试指标写回该算法的汇总记录。
        # predictions 保存数组供 bootstrap，prediction_frame 保存可持久化的逐条明细。
        predictions[name] = pred
        prediction_frame[f"{name}_prediction"] = pred
        prediction_frame[f"{name}_score"] = scores
    # 多数类基线只读取训练标签，零矩阵仅满足 sklearn 的输入形状要求。
    # 常量 0.5 分数表达没有排序信息，对同时含两类的测试集产生 AUC=0.5。
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train.label)
    # 例如训练标签中正面更多，基线对所有测试评论都预测 1，无需读取任何测试正文。
    baseline_prediction = baseline.predict(np.zeros((len(test), 1)))
    baseline_result = metrics(test.label, baseline_prediction, np.full(len(test), 0.5))
    baseline_result["model"] = "majority_baseline"
    predictions["majority_baseline"] = baseline_prediction
    # 固定已经训练好的模型，只重采样测试评论；所有模型使用相同索引作配对比较。
    uncertainty = bootstrap_accuracy(test.label, predictions, winner, config["bootstrap_repeats"], config["seed"])
    pred = predictions[winner]
    # 根据所选模型筛出真实误判，按真实标签各取前 40 条，正文截断到 1,500 字符。
    # 这是定性示例而非随机抽样；分组只用于展示，不影响性能指标。
    errors = test.loc[pred != test.label.to_numpy(), ["id", "label", "text"]].copy()
    # to_numpy 将真实标签转为按位置比较的数组，避免 pandas 索引对齐带来错位。
    # 同一错误掩码分别筛选 test 行与 pred 值，确保错误示例的正文、真实标签和预测匹配。
    errors["prediction"] = pred[pred != test.label.to_numpy()]
    errors["text"] = errors.text.str.slice(0, 1500)
    errors.groupby("label", group_keys=False).head(40).to_csv(paths.results / "error_examples.csv", index=False)
    # 每类不足 40 条时全部保留；head 保持输入顺序，本流程的源数据先前按 ID 排序，
    # 因此这些示例并非随机抽样，也不能用来估计某种错误机制的总体频率。
    # 评论长度分组用于事后观察；空组返回 None，避免输出没有样本支撑的准确率。
    lengths = test.text.str.split().str.len().to_numpy()
    length_analysis = []
    for label, lower, upper in [("<100 words", 0, 100), ("100–299 words", 100, 300), (">=300 words", 300, np.inf)]:
        # 左闭右开区间 [lower,upper) 互不重叠：恰好 100 词进入中组，300 词进入长组。
        # np.inf 让最后一组覆盖所有更长评论，mask 是长度为测试样本数的布尔数组。
        mask = (lengths >= lower) & (lengths < upper)
        length_analysis.append({"group": label, "n": int(mask.sum()),
                                "accuracy": float((pred[mask] == test.label.to_numpy()[mask]).mean()) if mask.any() else None})
    # 汇总真实指标、配置、数据审计和实际软件版本，报告直接消费这些产物，
    # 不重新训练，也不凭历史记录写死当前结果。
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
               "winner": winner, "train_rows": len(train), "test_rows": len(test),
               "models": records, "baseline": baseline_result, "uncertainty": uncertainty,
               "length_analysis": length_analysis, "feature_weights": explain_features(tuned[winner]),
               "data_audit": read_json(paths.results / "data_audit.json"),
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "packages": {name: version(name) for name in ["scikit-learn", "numpy", "pandas", "matplotlib", "joblib"]}}}
    write_json(paths.results / "metrics.json", summary)
    # metrics.json 是报告主输入，run_config.json 单独保留本次实际配置快照。
    # 若修改默认 experiment.json，旧快照仍说明旧结果使用了哪些参数。
    write_json(paths.results / "run_config.json", config)
    # gzip 的 mtime 固定为 0，消除压缩头中当前时间带来的差异，便于比较相同预测产物。
    prediction_frame.to_csv(paths.results / "test_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    metric_columns = ["model", "cv_macro_f1", "cv_std", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc",
                      "search_seconds", "refit_seconds", "predict_and_score_seconds", "vocabulary_size"]
    # 对比 CSV 是 metrics.json 的扁平视图，不含基线、区间或逐条预测，便于表格软件查看。
    pd.DataFrame(records)[metric_columns].to_csv(paths.results / "model_comparison.csv", index=False)
    return summary
