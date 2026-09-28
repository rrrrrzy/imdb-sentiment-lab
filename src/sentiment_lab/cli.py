"""Thin command layer: delegate work to the corresponding module."""
import argparse
import json
import logging
from pathlib import Path

from .config import Paths, load_config
from .data.collect import collect
from .data.prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="50K IMDb review classification course lab")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project directory (default: cwd)")
    parser.add_argument("--config", type=Path, help="JSON experiment config")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["collect", "prepare", "run"]:
        sub = commands.add_parser(command)
        sub.add_argument("--offline", action="store_true", help="Use verified archive cache; no network")
    commands.add_parser("train")
    commands.add_parser("report")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--text", required=True, help="English review text")
    server = commands.add_parser("serve")
    server.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    paths = Paths(args.root.resolve())
    config = load_config(args.config or paths.root / "configs" / "experiment.json")
    if args.command in {"collect", "prepare", "run"}:
        raw = collect(paths, offline=args.offline)
        if args.command != "collect":
            prepare(raw, paths, config["min_samples"])
    if args.command in {"train", "run"}:
        from .train import run_experiment
        summary = run_experiment(paths, config)
        print(f"CV-selected model: {summary['winner']}; {summary['test_rows']} test reviews")
    if args.command in {"report", "run"}:
        from .report import generate_report
        generate_report(paths)
        print(f"Report: {paths.reports / 'index.html'}")
    if args.command == "predict":
        from .predict import load_selected_model, predict_review
        name, model = load_selected_model(paths)
        print(json.dumps(predict_review(name, model, args.text), ensure_ascii=False, indent=2))
    if args.command == "serve":
        from .server import serve
        serve(paths, args.port)
