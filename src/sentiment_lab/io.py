"""Shared serialization helpers; no training or presentation logic here."""
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    # 输入为输出路径和可 JSON 序列化的值；函数直接覆盖文件，不返回序列化结果。
    # parents=True 补齐多级目录，exist_ok=True 允许重复生成报告/指标而不因目录已存在失败。
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保留中文，indent=2 便于人工复核及 Git diff；UTF-8 明确文件编码。
    # NumPy 标量/数组需在调用方转为 float/int/list，这里不提供隐式类型转换。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    # 项目调用方传入保存 JSON 对象的文件，因此返回值按 dict 使用。
    # 不吞掉文件缺失、编码或 JSON 格式错误，使上游阶段未完成的问题及时暴露。
    return json.loads(path.read_text(encoding="utf-8"))
