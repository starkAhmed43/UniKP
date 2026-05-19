import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_FEATURE_DIRNAME,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    atomic_save_csv,
    discover_split_jobs,
    feature_job_dir,
    normalize_threshold_args,
    summarize_seed_runs,
)


CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
FEATURE_SCRIPT = REPO_ROOT / "emulator_bench" / "build_split_features.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def run_stage(cmd, cwd, env=None):
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def maybe_cache(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--device",
        args.cache_device,
        "--protein_batch_size",
        str(args.protein_batch_size),
        "--protein_max_tokens",
        str(args.protein_max_tokens),
        "--smiles_batch_size",
        str(args.smiles_batch_size),
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.max_rows_per_split is not None:
        cmd.extend(["--max_rows_per_split", str(args.max_rows_per_split)])
    if args.keep_dot_smiles:
        cmd.append("--keep_dot_smiles")
    if args.cache_overwrite:
        cmd.append("--overwrite")
    run_stage(cmd, REPO_ROOT)


def maybe_build_features(args):
    if args.skip_feature_build:
        return
    cmd = [
        sys.executable,
        str(FEATURE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--feature_dirname",
        args.feature_dirname,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.max_rows_per_split is not None:
        cmd.extend(["--max_rows_per_split", str(args.max_rows_per_split)])
    if args.keep_dot_smiles:
        cmd.append("--keep_dot_smiles")
    if args.feature_overwrite:
        cmd.append("--overwrite")
    run_stage(cmd, REPO_ROOT)


def build_experiments(args, jobs):
    experiments = []
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / args.results_dirname
    for job in jobs:
        fdir = feature_job_dir(Path(args.base_dir), args.feature_dirname, job["split_group"], job["split_name"])
        for seed in args.seeds:
            out_dir = output_root / job["split_group"] / job["split_name"] / f"seed_{seed}"
            experiments.append(
                {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "feature_dir": fdir,
                    "out_dir": out_dir,
                    "seed": int(seed),
                }
            )
    return experiments


def train_command(exp, args):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--feature_dir",
        str(exp["feature_dir"]),
        "--out_dir",
        str(exp["out_dir"]),
        "--seed",
        str(exp["seed"]),
        "--random_state_mode",
        args.random_state_mode,
    ]
    if args.model_n_jobs is not None:
        cmd.extend(["--model_n_jobs", str(args.model_n_jobs)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_one(exp, args, worker_index):
    metric_path = exp["out_dir"] / "final_results_test.csv"
    model_path = exp["out_dir"] / "model.joblib"
    if metric_path.exists() and model_path.exists() and not args.overwrite:
        return {
            "status": "skipped_exists",
            "worker_index": worker_index,
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
            "out_dir": str(exp["out_dir"]),
        }

    env = os.environ.copy()
    if args.model_n_jobs is not None and args.model_n_jobs > 0:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(args.model_n_jobs)

    cmd = train_command(exp, args)
    if args.cpu_affinity:
        affinity = args.cpu_affinity[worker_index % len(args.cpu_affinity)]
        cmd = ["taskset", "-c", affinity] + cmd
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return {
        "status": "completed",
        "worker_index": worker_index,
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
        "out_dir": str(exp["out_dir"]),
    }


def run_parallel(experiments, args):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)
    rows = []
    lock = threading.Lock()

    def worker(worker_index):
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                row = run_one(exp, args, worker_index)
            except Exception as exc:
                row = {
                    "status": "failed",
                    "worker_index": worker_index,
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                    "out_dir": str(exp["out_dir"]),
                    "error": str(exc),
                }
            with lock:
                rows.append(row)
            work_queue.task_done()

    threads = []
    for worker_index in range(args.num_workers):
        thread = threading.Thread(target=worker, args=(worker_index,), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return rows


def collect_metrics(rows):
    collected = []
    for row in rows:
        out = dict(row)
        run_dir = Path(row["out_dir"])
        for split in ("train", "val", "test"):
            path = run_dir / f"final_results_{split}.csv"
            if path.exists():
                metrics = pd.read_csv(path).iloc[0].to_dict()
                for key, value in metrics.items():
                    out[f"{split}_{key}"] = value
        collected.append(out)
    return collected


def main():
    parser = argparse.ArgumentParser(description="Parallel multirun UniKP retraining on original ExtraTrees settings.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--feature_dirname", type=str, default=DEFAULT_FEATURE_DIRNAME)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--model_n_jobs", type=int, default=None)
    parser.add_argument("--cpu_affinity", nargs="+", default=None, help="Optional CPU sets, one per worker slot, e.g. 0-15 16-31.")
    parser.add_argument("--random_state_mode", choices=["seed", "none"], default="seed")
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--protein_batch_size", type=int, default=4)
    parser.add_argument("--protein_max_tokens", type=int, default=4096)
    parser.add_argument("--smiles_batch_size", type=int, default=512)
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--keep_dot_smiles", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--skip_feature_build", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--feature_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache(args)
    maybe_build_features(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    experiments = build_experiments(args, jobs)
    rows = collect_metrics(run_parallel(experiments, args))

    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / args.results_dirname
    output_root.mkdir(parents=True, exist_ok=True)
    runs_path = output_root / "retrain_summary_runs.csv"
    atomic_save_csv(runs_path, pd.DataFrame(rows))
    metric_cols = [col for col in pd.DataFrame(rows).columns if col.startswith("test_")]
    summarize_seed_runs(rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(
        output_root / "retrain_summary_by_split.csv",
        index=False,
    )
    summarize_seed_runs(rows, ["split_group"], metric_cols).to_csv(
        output_root / "retrain_summary_by_split_group.csv",
        index=False,
    )
    print(f"Saved retrain summary to {runs_path}", flush=True)


if __name__ == "__main__":
    main()
