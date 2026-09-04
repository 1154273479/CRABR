from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ["dice", "iou", "jaccard", "precision", "recall", "f1", "bf_score", "hd95"]
CORE_METRICS = ["dice", "iou", "bf_score", "hd95"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SOTA benchmark results")
    parser.add_argument("--output-root", type=str, default="sota_runs")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--markdown", type=str, default="")
    parser.add_argument("--dataset", type=str, default="", help="Filter by dataset name")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_from_result(path: Path, root: Path) -> dict[str, Any] | None:
    data = _load_json(path)
    metrics = data.get("overall", data)
    if not all(key in metrics for key in ("dice", "iou", "bf_score", "hd95")):
        return None

    rel_parts = path.relative_to(root).parts
    dataset = Path(str(data.get("dataset_config", rel_parts[0] if rel_parts else ""))).stem
    run_name = rel_parts[1] if len(rel_parts) > 1 else path.parent.parent.name
    model = data.get("model_name", run_name.split("_seed")[0])
    seed = data.get("seed", "")
    row = {
        "dataset": dataset,
        "model": model,
        "run": run_name,
        "seed": seed,
        "path": str(path),
        "per_class": data.get("per_class", {}),
    }
    for key in METRIC_KEYS:
        row[key] = float(metrics.get(key, 0.0))
    return row


def collect_rows(root: Path, dataset_filter: str = "") -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/**/results/*.json")) + sorted(root.glob("*/**/checkpoints/best_metrics.json"))
    rows = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        row = _row_from_result(path, root)
        if row is not None:
            if dataset_filter and dataset_filter not in row["dataset"]:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], -r["dice"], r["hd95"], str(r["model"])))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "model", "run", "seed", *METRIC_KEYS, "path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in columns})


def _rank_symbol(rank: int) -> str:
    if rank == 1:
        return " **#1**"
    if rank == 2:
        return " *#2*"
    return ""


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# SOTA Benchmark Summary\n"]

    datasets = sorted(set(r["dataset"] for r in rows))
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue

        lines.append(f"## Dataset: {ds}\n")

        # --- Overall table ---
        lines.append("### Overall Metrics\n")
        lines.append("| Rank | Model | Params | Dice | IoU | BF Score | HD95 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")

        for rank, row in enumerate(ds_rows, 1):
            model_name = row["model"]
            if model_name == "erga":
                model_name = "**ERGA (Ours)**"
            lines.append(
                f"| {rank} | {model_name} | - | "
                f"{row['dice']:.5f} | {row['iou']:.5f} | {row['bf_score']:.5f} | {row['hd95']:.3f} |"
            )

        # --- Per-class table ---
        class_names: list[str] = []
        for row in ds_rows:
            if row.get("per_class"):
                class_names = list(row["per_class"].keys())
                break

        if class_names:
            lines.append("\n### Per-Class Metrics\n")
            header = "| Model |"
            sep = "|---|"
            for cn in class_names:
                header += f" {cn} Dice | {cn} BF | {cn} HD95 |"
                sep += "---:|---:|---:|"
            lines.append(header)
            lines.append(sep)

            for row in ds_rows:
                pc = row.get("per_class", {})
                model_name = row["model"]
                if model_name == "erga":
                    model_name = "**ERGA (Ours)**"
                line = f"| {model_name} |"
                for cn in class_names:
                    cm = pc.get(cn, {})
                    line += f" {cm.get('dice', 0):.4f} | {cm.get('bf_score', 0):.4f} | {cm.get('hd95', 0):.2f} |"
                lines.append(line)

        # --- Best metrics highlight ---
        lines.append("\n### Best Metrics\n")
        for metric in CORE_METRICS:
            if metric == "hd95":
                best_row = min(ds_rows, key=lambda r: r[metric])
            else:
                best_row = max(ds_rows, key=lambda r: r[metric])
            lines.append(f"- **Best {metric.upper()}**: {best_row['model']} = {best_row[metric]:.5f}")

        lines.append("\n---\n")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown summary: {path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    datasets = sorted(set(r["dataset"] for r in rows))
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}  ({len(ds_rows)} models)")
        print(f"{'='*60}")
        print(f"{'Rank':<5} {'Model':<20} {'Dice':<8} {'IoU':<8} {'BF':<8} {'HD95':<8}")
        print("-" * 60)
        for rank, row in enumerate(ds_rows, 1):
            marker = " <--" if row["model"] == "erga" else ""
            print(
                f"{rank:<5} {row['model']:<20} {row['dice']:.5f}  {row['iou']:.5f}  "
                f"{row['bf_score']:.5f}  {row['hd95']:.3f}{marker}"
            )


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    rows = collect_rows(root, args.dataset)
    csv_path = Path(args.csv) if args.csv else root / "sota_summary.csv"
    md_path = Path(args.markdown) if args.markdown else root / "sota_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print_summary(rows)
    print(f"\nCollected {len(rows)} result files")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
