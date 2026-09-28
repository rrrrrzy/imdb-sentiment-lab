"""Clean HTML and whitespace while retaining negations; audit duplicates."""
import hashlib
import html
import re

import pandas as pd

from ..config import Paths
from ..io import write_json


def clean_text(value: str) -> str:
    # 输入单条评论字符串，返回规范化字符串；既不分词，也不输出 TF-IDF 特征。
    # 例：'I am NOT<br /> happy &amp; satisfied!' → 'i am not happy & satisfied!'。
    # 训练、数据去重与在线推理共用此函数，保证文本规范化规则一致。
    # 用空格替换标签，避免 good<br>movie 被连接成 goodmovie；再解码 &amp; 等实体。
    value = re.sub(r"<[^>]*>", " ", value)
    # 顺序是先去除原文本中的标签，再解码实体；这是本项目统一采用的处理约定。
    value = html.unescape(value).lower()
    # 仅合并空白并去掉首尾空格，不删除 not/no 等否定词，也不按情感标签改写正文。
    return re.sub(r"\s+", " ", value).strip()


def audit_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # 输入表至少需要 id、split、label、text 四列，采集表还带 rating 等来源信息。
    # 输出 (clean, audit)：clean 保留原列并增加 text_hash；audit 记录各步移除数量、
    # 每个划分的类别数和词数统计。函数复制输入，避免修改调用方持有的原始表。
    # ID 用于追踪逐条预测，必须唯一；标签语义固定为 0=负面、1=正面。
    if raw["id"].duplicated().any() or not set(raw["label"].unique()) <= {0, 1}:
        raise ValueError("IDs must be unique and labels must be binary")
    frame = raw.copy()
    frame["text"] = frame["text"].map(clean_text)
    # map 逐条调用字符串清洗；eq('') 产生布尔掩码，sum 将 True 按 1 计数。
    # 有些评论原本非空但只有 HTML 标签，必须在清洗后再判断是否为空。
    audit = {"original_rows": len(frame), "empty_removed": int(frame.text.eq("").sum()),
             "within_split": {}}
    frame = frame.loc[frame.text.ne("")].copy()
    # 在清洗后计算哈希，使仅 HTML、大小写或空白不同的评论被视为同一正文。
    frame["text_hash"] = frame.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    # parts 的结构为 {'train': DataFrame, 'test': DataFrame}，使后续重复检查方向明确。
    parts = {}
    for split in ("train", "test"):
        # 各划分独立处理标签冲突，训练集清洗不依赖测试标签。
        subset = frame.loc[frame.split.eq(split)].copy()
        conflicts = subset.groupby("text_hash").label.nunique()
        # groupby 按正文分组，nunique 计算每组不同标签的数量：
        # [1,1] → 1（同标签重复），[0,1] → 2（标签冲突），冲突组要全删而非保留第一条。
        # 同一正文对应多个标签时全部删除，不能任意挑一条作为“正确答案”。
        conflicts = set(conflicts[conflicts > 1].index)
        ambiguous = subset.text_hash.isin(conflicts)
        # isin 得到与 subset 行索引对应的布尔 Series；loc[~ambiguous] 中 ~ 是逐项取反。
        audit["within_split"][split] = {"conflicting_rows_removed": int(ambiguous.sum())}
        subset = subset.loc[~ambiguous]
        # 排除冲突后，其余同正文记录标签一致，按当前顺序保留第一条即可。
        audit["within_split"][split]["duplicate_rows_removed"] = int(subset.text_hash.duplicated().sum())
        parts[split] = subset.drop_duplicates("text_hash")
    # Honor the original test boundary; never move test reviews into training.
    # 跨划分只比较正文哈希，并仅删除测试重复，避免“见过答案”抬高测试分数。
    # 保留原划分，不合并后随机重切；清理后的测试集数量会小于原始 25,000 条。
    overlap = parts["test"].text_hash.isin(set(parts["train"].text_hash))
    audit["cross_split_test_duplicates_removed"] = int(overlap.sum())
    parts["test"] = parts["test"].loc[~overlap]
    # concat 按 train、test 顺序合并；ignore_index=True 重建连续行号。
    # 影评身份由 id 表示，DataFrame 行号只是处理位置，不能替代来源 ID。
    clean = pd.concat(parts.values(), ignore_index=True)
    audit["clean_rows"] = len(clean)
    # 同步输出类别数量与长度分布，供报告展示；这些统计不用于拟合特征或选模型。
    audit["counts"] = {split: {str(label): int(count) for label, count in part.label.value_counts().sort_index().items()}
                       for split, part in parts.items()}
    # JSON 对象的键是字符串，所以类别键转成 '0'/'1'；计数转 Python int 以支持序列化。
    # str.split().str.len() 按空白计算词数；p95 为第 95 百分位，不是准确率区间端点。
    audit["word_length"] = {
        split: {"median": float(part.text.str.split().str.len().median()),
                "mean": float(part.text.str.split().str.len().mean()),
                "p95": float(part.text.str.split().str.len().quantile(0.95))}
        for split, part in parts.items()}
    # 最后再次断言两划分没有完全相同的规范化正文，保护下游训练流程。
    # 此检查不覆盖近似复述或同一电影评论间的相关性。
    if set(parts["train"].text_hash) & set(parts["test"].text_hash):
        raise AssertionError("Text leakage across train/test")
    return clean, audit


def prepare(raw: pd.DataFrame, paths: Paths, min_samples: int) -> pd.DataFrame:
    # 阶段入口：接收原始表、项目路径和清洗后最低样本量；返回已清洗表，同时写两个文件。
    # reviews.csv.gz 供 train 读取，data_audit.json 供 train 汇总及 report 展示。
    clean, audit = audit_and_clean(raw)
    if len(clean) < min_samples:
        raise ValueError(f"At least {min_samples} clean reviews are required")
    paths.processed.mkdir(parents=True, exist_ok=True)
    # 保存可供训练直接读取的压缩 CSV，另存审计 JSON 记录每一步移除数量。
    # 最低样本量检查作用于清洗后的数量，不能用原始 50,000 条代替。
    clean.to_csv(paths.processed / "reviews.csv.gz", index=False, compression="gzip")
    write_json(paths.results / "data_audit.json", audit)
    return clean
