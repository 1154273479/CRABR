from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOTA benchmark matrix")
    parser.add_argument("--matrix", type=str, default="benchmark/benchmark_matrix.yaml")
    parser.add_argument("--stage", choices=["train", "infer", "all", "summary"], default="all")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-script", type=str, default="")
    return parser.parse_args()


def _item_name(item: str | dict[str, Any]) -> str:
    return item if isinstance(item, str) else str(item["name"])


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_commands(matrix: dict[str, Any], args: argparse.Namespace) -> list[list[str]]:
    output_root = args.output_root or matrix.get("output_root", "sota_runs")
    datasets = matrix["datasets"]
    models = matrix["models"]
    seeds = matrix.get("seeds", [42])
    commands: list[list[str]] = []

    for dataset in datasets:
        config = dataset["config"] if isinstance(dataset, dict) else dataset
        for model in models:
            model_name = _item_name(model)
            for seed in seeds:
                base = [
                    sys.executable,
                    "-m",
                    "benchmark.train_sota",
                    "--config",
                    str(config),
                    "--model",
                    model_name,
                    "--seed",
                    str(seed),
                    "--output-root",
                    str(output_root),
                ]
                if args.gpu_ids:
                    base.extend(["--gpu-ids", args.gpu_ids])
                infer = [
                    sys.executable,
                    "-m",
                    "benchmark.infer_sota",
                    "--config",
                    str(config),
                    "--model",
                    model_name,
                    "--seed",
                    str(seed),
                    "--output-root",
                    str(output_root),
                    "--split",
                    args.split,
                    "--no-save-images",
                ]
                if args.stage in ("train", "all"):
                    commands.append(base)
                if args.stage in ("infer", "all"):
                    commands.append(infer)

    if args.stage == "summary":
        commands.append(
            [
                sys.executable,
                "-m",
                "benchmark.compare_results",
                "--output-root",
                str(output_root),
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    with Path(args.matrix).open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f)
    commands = build_commands(matrix, args)

    if args.write_script:
        script_path = Path(args.write_script)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("\n".join(_quote_cmd(cmd) for cmd in commands) + "\n", encoding="utf-8")
        print(f"Wrote commands to: {script_path}")

    failed: list[str] = []
    for cmd in commands:
        print(_quote_cmd(cmd))
        if not args.dry_run:
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                failed.append(_quote_cmd(cmd))
                print(f"[WARN] Command failed (exit {ret.returncode}), skipping.")

    if failed:
        print(f"\n[SUMMARY] {len(failed)} command(s) failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
