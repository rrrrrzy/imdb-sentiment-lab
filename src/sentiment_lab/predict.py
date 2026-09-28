"""Inference uses the persisted pipeline and exactly the same text cleanup."""
import joblib

from .config import Paths
from .data.prepare import clean_text
from .evaluation import continuous_scores
from .io import read_json


def load_selected_model(paths: Paths):
    # 输入项目路径，返回 (算法名, 已拟合 Pipeline) 元组。
    # selection.json 与 models/<算法名>.joblib 必须来自同一次有效训练，缺失时不会自动重训。
    # 按训练阶段冻结的 CV 选择加载模型，不根据测试分数在推理时重新挑选。
    selection = read_json(paths.results / "selection.json")
    name = selection["winner"]
    # joblib/pickle must only be loaded from this project's trusted training output.
    # joblib 反序列化可能执行代码，只能读取本项目可信训练输出；文件包含完整 Pipeline。
    return name, joblib.load(paths.models / f"{name}.joblib")


def predict_review(name: str, model, text: str) -> dict:
    # 输入模型名称、已加载 Pipeline 和原始字符串；返回可直接 JSON 化的单条结果。
    # 返回字段：model 标识算法，label 为 0/1，score 为正面倾向，score_type 说明量纲。
    # 输入不合格抛 ValueError，由 CLI 显示异常或由 server 转为 HTTP 400。
    # 字符数检查针对原始输入，先验证类型，兼容接口中缺失 text 或传入非字符串的情况。
    if not isinstance(text, str) or not 2 <= len(text) <= 10000:
        raise ValueError("Please enter 2–10000 characters of English review text")
    cleaned = clean_text(text)
    # 长度合法的输入也可能只有空格或 HTML 标签，因此清洗后仍需检查空正文。
    if not cleaned:
        raise ValueError("No review text remains after HTML cleaning")
    # 只用训练时的词表做 transform，不能对新输入 fit，否则会改变模型特征空间。
    # sklearn 要求输入为一批文本，所以单条评论也包装成长度为 1 的列表。
    features = model.named_steps["tfidf"].transform([cleaned])
    # 单条输入得到形状 (1,V) 的稀疏矩阵，V 为训练时词表大小。
    # 例如输入全是词表外的新词，矩阵全零；至少出现一个已知词/词组才通过下面检查。
    # nnz 是稀疏矩阵的非零元素数；为 0 表示没有已知特征，此时预测可能只由截距
    # 或类别先验决定，缺乏文本依据。该检查判断词表覆盖，并非严格的语言识别器。
    if features.nnz == 0:
        raise ValueError("No known English features found; please use an English movie review")
    # 返回标签及连续分数，numpy 标量转换为 Python 标量以支持 JSON 序列化。
    # LR/SVM 的决策分数与 NB 的正面概率含义不同，显式返回 score_type 供界面说明。
    # predict([cleaned])[0] 取批预测的第一个标签；打分同样取第一个元素。
    # 前面的 transform 只用于检查覆盖度，返回结果仍由完整 Pipeline 处理 cleaned。
    # SVM 分数大于 0 通常对应正面，但绝对值不是“正确率”；NB 概率也未做额外校准。
    return {"model": name, "label": int(model.predict([cleaned])[0]),
            "score": float(continuous_scores(model, [cleaned])[0]),
            "score_type": "decision score" if hasattr(model, "decision_function") else "positive probability"}
