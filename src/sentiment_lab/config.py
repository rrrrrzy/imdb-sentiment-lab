"""Explicit configuration and project paths."""
from dataclasses import dataclass
from pathlib import Path

from .io import read_json


@dataclass(frozen=True)
class Paths:
    # 输入 root 为项目根目录的 Path；例如 Paths(Path('/work/imdb-sentiment-lab'))。
    # dataclass 自动生成构造方法，frozen=True 禁止实例创建后重新赋值 root。
    # 各目录通过 property 按需计算，不存储重复路径，也不会在访问时创建目录。
    # Path 的 / 运算符拼接目录，调用方可继续写 paths.results / 'metrics.json'。
    root: Path

    @property
    def raw(self) -> Path:
        # 原始归档、发布页快照和 provenance.json 放在这里，供缓存校验与离线重跑。
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        # 清洗后的 reviews.csv.gz 是训练输入；保留正文、原始划分和正文哈希。
        return self.root / "data" / "processed"

    @property
    def results(self) -> Path:
        # 轻量实验记录：数据审计、逐折 CV、模型选择、指标、测试预测及错误示例。
        return self.root / "artifacts" / "results"

    @property
    def models(self) -> Path:
        # 大型 joblib Pipeline 存放目录，与可阅读的结果记录分开管理。
        return self.root / "artifacts" / "models"

    @property
    def reports(self) -> Path:
        # HTML、Markdown、静态网页资源及 figures 子目录的统一输出位置。
        return self.root / "reports"


def load_config(path: Path) -> dict:
    # 输入：实验 JSON 路径；输出：供 CLI、模型工厂和训练模块共用的配置字典。
    # seed/cv_folds/n_jobs 控制实验执行，features 配置 TF-IDF，models 配置参数网格。
    # 此处只检查下面三组实验约束，未做完整 schema 验证；缺键等问题会继续抛出异常。
    config = read_json(path)
    # CV 至少需要两折；清洗后最低样本量不得低于课程规定的 10,000 条。
    if config["cv_folds"] < 2 or config["min_samples"] < 10000:
        raise ValueError("cv_folds must be >= 2 and min_samples must be >= 10000")
    # 集合比较不依赖 JSON 键顺序，但要求三种支持算法恰好全部存在。
    if set(config["models"]) != {"multinomial_nb", "logistic_regression", "linear_svm"}:
        raise ValueError("The experiment requires all three supported models")
    # 重采样过少会使区间端点不稳定；默认 1,000 次，此处仅设置最低 100 次门槛。
    if config["bootstrap_repeats"] < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    return config
