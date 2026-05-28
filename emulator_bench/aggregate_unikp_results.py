import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_RESULTS_DIRNAME, atomic_save_csv, summarize_seed_runs


DEFAULT_TASK_ROOTS = {
    "kcat": DEFAULT_BASE_DIR.parent / "UniKP_Kcat" / DEFAULT_RESULTS_DIRNAME,
    "km": DEFAULT_BASE_DIR.parent / "UniKP_Km" / DEFAULT_RESULTS_DIRNAME,
}


def infer_task(path: Path) -> str:
    text = str(path).lower()
    if "unikp_kcat" in text or "kcat" in text:
        return "kcat"
    if "unikp_km" in text or "/km" in text:
        return "km"
    return Path(path).name


def _parse_run_parts(root: Path, run_dir: Path, task: str = None):
    row = {"run_dir": str(run_dir)}
    if task:
        row["task"] = task
    try:
        parts = run_dir.relative_to(root).parts
    except ValueError:
        return row

    if len(parts) >= 3:
        row["split_group"] = parts[-3]
        row["split_name"] = parts[-2]
        seed_part = parts[-1]
        row["seed"] = seed_part.replace("seed_", "") if seed_part.startswith("seed_") else seed_part
    return row


def _load_json(path: Path):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _collect_one_run(root: Path, run_dir: Path, task: str = None):
    row = _parse_run_parts(root, run_dir, task=task)

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        for key in ("elapsed_seconds", "feature_dir", "out_dir", "seed", "model_n_jobs"):
            if key in summary and key not in row:
                row[key] = summary[key]

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = _load_json(config_path)
        params = config.get("extra_trees_params", {})
        for key, value in params.items():
            row[f"param_{key}"] = value

    for split in ("train", "val", "test"):
        metrics_path = run_dir / f"final_results_{split}.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        for key, value in metrics.items():
            row[f"{split}_{key}"] = value
    return row


def collect_runs(root: Path, task: str = None):
    root = Path(root)
    candidate_dirs = set()
    for summary_path in root.glob("**/run_summary.json"):
        candidate_dirs.add(summary_path.parent)
    for metrics_path in root.glob("**/final_results_test.csv"):
        candidate_dirs.add(metrics_path.parent)

    rows = []
    for run_dir in sorted(candidate_dirs):
        row = _collect_one_run(root, run_dir, task=task)
        if any(key.startswith("test_") for key in row):
            rows.append(row)
    return rows


def collect_many_roots(roots):
    rows = []
    for task, root in roots:
        root = Path(root)
        if not root.exists():
            print(f"Skipping missing result root for {task}: {root}")
            continue
        rows.extend(collect_runs(root, task=task))
    return rows


def metric_columns(frame: pd.DataFrame):
    excluded = {"test_split", "test_seed", "test_model_n_jobs"}
    return [col for col in frame.columns if col.startswith("test_") and col not in excluded]


def main():
    parser = argparse.ArgumentParser(description="Aggregate UniKP emulator_bench result directories.")
    parser.add_argument("--root", type=str, default=None, help="Aggregate one result root. If omitted, aggregates UniKP_Kcat and UniKP_Km.")
    parser.add_argument("--roots", nargs="+", default=None, help="Aggregate multiple result roots. Task labels are inferred from path names.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(DEFAULT_TASK_ROOTS), default=None, help="Default task roots to aggregate.")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.root and args.roots:
        raise ValueError("Use either --root or --roots, not both.")

    if args.root:
        root = Path(args.root)
        rows = collect_runs(root, task=infer_task(root))
        out_dir = Path(args.out_dir) if args.out_dir else root / "aggregated"
    else:
        if args.roots:
            root_items = [(infer_task(Path(root)), Path(root)) for root in args.roots]
        else:
            tasks = args.tasks or sorted(DEFAULT_TASK_ROOTS)
            root_items = [(task, DEFAULT_TASK_ROOTS[task]) for task in tasks]
        rows = collect_many_roots(root_items)
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_BASE_DIR.parent / "UniKP_aggregated_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(rows)
    atomic_save_csv(out_dir / "aggregated_run_summaries.csv", runs)

    if not runs.empty:
        metric_cols = metric_columns(runs)
        if {"task", "split_group", "split_name"}.issubset(runs.columns):
            atomic_save_csv(
                out_dir / "summary_by_task_split.csv",
                summarize_seed_runs(rows, ["task", "split_group", "split_name"], metric_cols),
            )
        if {"task", "split_group"}.issubset(runs.columns):
            atomic_save_csv(
                out_dir / "summary_by_task_split_group.csv",
                summarize_seed_runs(rows, ["task", "split_group"], metric_cols),
            )
        if {"task"}.issubset(runs.columns):
            atomic_save_csv(
                out_dir / "summary_by_task.csv",
                summarize_seed_runs(rows, ["task"], metric_cols),
            )
        if {"split_group", "split_name"}.issubset(runs.columns):
            atomic_save_csv(
                out_dir / "summary_by_split.csv",
                summarize_seed_runs(rows, ["split_group", "split_name"], metric_cols),
            )
        if {"split_group"}.issubset(runs.columns):
            atomic_save_csv(
                out_dir / "summary_by_split_group.csv",
                summarize_seed_runs(rows, ["split_group"], metric_cols),
            )
    print(f"Saved aggregation outputs to {out_dir}")


if __name__ == "__main__":
    main()
