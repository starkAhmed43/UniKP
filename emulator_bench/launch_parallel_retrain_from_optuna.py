import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import optuna
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_FEATURE_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    atomic_save_csv,
    discover_split_jobs,
    feature_job_dir,
    normalize_threshold_args,
    summarize_seed_runs,
)


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def load_best_hparams(args):
    if args.hparams_json:
        with open(args.hparams_json, "r") as handle:
            payload = json.load(handle)
        return payload.get("best_hparams", payload)
    if not args.storage:
        raise ValueError("Provide --hparams_json or --storage.")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    return dict(study.best_params)


def write_hparams_json(output_root: Path, hparams: dict) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "resolved_best_hparams.json"
    with open(path, "w") as handle:
        json.dump({"best_hparams": hparams}, handle, indent=2, sort_keys=True)
    return path


def build_experiments(args, jobs):
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / "retrain_from_optuna"
    experiments = []
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
    return output_root, experiments


def train_command(exp, args, hparams_path):
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
        "--hparams_json",
        str(hparams_path),
    ]
    if args.model_n_jobs is not None:
        cmd.extend(["--model_n_jobs", str(args.model_n_jobs)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_parallel(experiments, args, hparams_path):
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
                metric_path = exp["out_dir"] / "final_results_test.csv"
                model_path = exp["out_dir"] / "model.joblib"
                if metric_path.exists() and model_path.exists() and not args.overwrite:
                    row = {"status": "skipped_exists"}
                else:
                    env = os.environ.copy()
                    if args.model_n_jobs is not None and args.model_n_jobs > 0:
                        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                            env[key] = str(args.model_n_jobs)
                    cmd = train_command(exp, args, hparams_path)
                    if args.cpu_affinity:
                        affinity = args.cpu_affinity[worker_index % len(args.cpu_affinity)]
                        cmd = ["taskset", "-c", affinity] + cmd
                    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
                    row = {"status": "completed"}
                row.update(
                    {
                        "worker_index": worker_index,
                        "split_group": exp["split_group"],
                        "split_name": exp["split_name"],
                        "difficulty": exp["difficulty"],
                        "seed": exp["seed"],
                        "out_dir": str(exp["out_dir"]),
                    }
                )
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
    parser = argparse.ArgumentParser(description="Parallel UniKP retraining from Optuna best ExtraTrees hyperparameters.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--feature_dirname", type=str, default=DEFAULT_FEATURE_DIRNAME)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--model_n_jobs", type=int, default=None)
    parser.add_argument("--cpu_affinity", nargs="+", default=None)
    parser.add_argument("--random_state_mode", choices=["seed", "none"], default="seed")
    parser.add_argument("--study_name", type=str, default="unikp_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    output_root, experiments = build_experiments(args, jobs)
    hparams_path = write_hparams_json(output_root, load_best_hparams(args))
    rows = collect_metrics(run_parallel(experiments, args, hparams_path))
    atomic_save_csv(output_root / "retrain_summary_runs.csv", pd.DataFrame(rows))
    metric_cols = [col for col in pd.DataFrame(rows).columns if col.startswith("test_")]
    summarize_seed_runs(rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(
        output_root / "retrain_summary_by_split.csv",
        index=False,
    )
    summarize_seed_runs(rows, ["split_group"], metric_cols).to_csv(
        output_root / "retrain_summary_by_split_group.csv",
        index=False,
    )
    print(f"Saved Optuna retrain summary to {output_root}", flush=True)


if __name__ == "__main__":
    main()
