"""Thin command layer: delegate work to the corresponding module."""
import argparse
import json
import logging
from pathlib import Path

from .config import Paths, load_config
from .data.collect import collect
from .data.prepare import prepare


def main() -> None:
    # 输入来自进程命令行参数，函数没有返回数据，阶段结果写文件或打印到标准输出。
    # 例：sentiment-lab --root /work/lab run --offline；root/config 属于全局参数，
    # run/offline 属于子命令。项目路径默认当前工作目录，而非源码所在目录。
    # 全局路径/配置参数定义在主解析器，需放在子命令前；子命令只负责各阶段选项。
    # --offline 只提供给需要采集数据的 collect/prepare/run。
    parser = argparse.ArgumentParser(description="50K IMDb review classification course lab")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project directory (default: cwd)")
    parser.add_argument("--config", type=Path, help="JSON experiment config")
    commands = parser.add_subparsers(dest="command", required=True)
    # dest='command' 将子命令名保存到 args.command；required=True 禁止省略子命令。
    for command in ["collect", "prepare", "run"]:
        sub = commands.add_parser(command)
        sub.add_argument("--offline", action="store_true", help="Use verified archive cache; no network")
        # store_true 将出现该开关解释为 True，未传时为 False，不需要显式填写布尔值。
    commands.add_parser("train")
    commands.add_parser("report")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--text", required=True, help="English review text")
    server = commands.add_parser("serve")
    server.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    paths = Paths(args.root.resolve())
    # resolve 将项目根目录转为绝对路径；自定义 config 路径仍按命令行传入值读取。
    # load_config 在所有子命令前执行，因此 report/predict/serve 也需要有效配置文件。
    config = load_config(args.config or paths.root / "configs" / "experiment.json")
    # 这些独立 if 让 run 依次经过采集+清洗、训练、报告三个阶段。
    # prepare 也先调用 collect：优先读取校验后的本地归档，再将原始评论交给清洗模块。
    if args.command in {"collect", "prepare", "run"}:
        raw = collect(paths, offline=args.offline)
        if args.command != "collect":
            prepare(raw, paths, config["min_samples"])
    if args.command in {"train", "run"}:
        # 按命令延迟导入训练/绘图/推理模块，采集命令不必提前加载这些模块。
        from .train import run_experiment
        summary = run_experiment(paths, config)
        print(f"CV-selected model: {summary['winner']}; {summary['test_rows']} test reviews")
    if args.command in {"report", "run"}:
        # report 直接读取已有训练结果；run 进入这里时，前面的阶段已按顺序成功完成。
        # 阶段异常会中断流程，不能继续生成看似成功的新报告。
        from .report import generate_report
        generate_report(paths)
        print(f"Report: {paths.reports / 'index.html'}")
    if args.command == "predict":
        # 命令行推理与网页接口共用相同模型加载和输入校验函数，避免两套预测逻辑。
        from .predict import load_selected_model, predict_review
        name, model = load_selected_model(paths)
        print(json.dumps(predict_review(name, model, args.text), ensure_ascii=False, indent=2))
        # 终端输出标准 JSON 对象，保留中文并缩进，便于人工查看或由其他程序读取。
    if args.command == "serve":
        from .server import serve
        serve(paths, args.port)
