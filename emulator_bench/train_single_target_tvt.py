import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_FEATURE_DIRNAME,
    DEFAULT_RESULTS_DIRNAME,
    atomic_save_csv,
    atomic_save_json,
    configure_cpu_environment,
    feature_job_dir,
    original_extra_trees_params,
    regression_metrics,
    resolve_single_split_job,
    set_seed,
)


def load_feature_split(feature_dir: Path, split: str):
    matrix_path = Path(feature_dir) / f"{split}.npz"
    metadata_path = Path(feature_dir) / f"{split}_metadata.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    with np.load(matrix_path, allow_pickle=False) as data:
        x = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.float32, copy=False)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(y):
        raise ValueError(f"Metadata/target length mismatch for {split}: {len(metadata)} != {len(y)}")
    return x, y, metadata


def save_predictions(path: Path, metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    frame = metadata.copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    atomic_save_csv(path, frame)


def resolve_feature_dir(args) -> Path:
    if args.feature_dir:
        return Path(args.feature_dir)
    job = resolve_single_split_job(Path(args.base_dir), split_group=args.split_group, threshold=args.threshold)
    return feature_job_dir(Path(args.base_dir), args.feature_dirname, job["split_group"], job["split_name"])


def default_out_dir(args, feature_dir: Path) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if args.split_group:
        split_name = args.threshold or args.split_group
        return Path(args.base_dir) / args.results_dirname / args.split_group / split_name / f"seed_{args.seed}"
    return feature_dir / args.results_dirname / f"seed_{args.seed}"


def load_hparams_json(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r") as handle:
        payload = json.load(handle)
    return payload.get("best_hparams", payload)


def none_if_negative(value):
    if value is None:
        return None
    return None if int(value) < 0 else int(value)


def build_tree_overrides(args) -> dict:
    raw = load_hparams_json(args.hparams_json)
    overrides = {}
    keys = [
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "min_weight_fraction_leaf",
        "max_features",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "bootstrap",
        "oob_score",
        "verbose",
        "warm_start",
        "ccp_alpha",
        "max_samples",
    ]
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
        elif key in raw:
            overrides[key] = raw[key]
    for key in ("max_depth", "max_leaf_nodes", "max_samples"):
        if key in overrides and overrides[key] == "none":
            overrides[key] = None
    for key in ("max_depth", "max_leaf_nodes"):
        if key in overrides:
            overrides[key] = none_if_negative(overrides[key])
    for key in ("min_samples_split", "min_samples_leaf"):
        if key in overrides and isinstance(overrides[key], float) and overrides[key] >= 1 and overrides[key].is_integer():
            overrides[key] = int(overrides[key])
    return overrides


def maybe_threadpool_limits(model_n_jobs):
    if model_n_jobs is None or int(model_n_jobs) <= 0:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(model_n_jobs))
    except Exception:
        return nullcontext()


def main():
    parser = argparse.ArgumentParser(description="Train one UniKP ExtraTrees model on cached train/val/test feature matrices.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--feature_dirname", type=str, default=DEFAULT_FEATURE_DIRNAME)
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--split_group", type=str, default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--random_state_mode", choices=["seed", "none"], default="seed")
    parser.add_argument("--model_n_jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--criterion", type=str, default=None)
    parser.add_argument("--max_depth", type=int, default=None, help="Use -1 for None.")
    parser.add_argument("--min_samples_split", type=float, default=None)
    parser.add_argument("--min_samples_leaf", type=float, default=None)
    parser.add_argument("--min_weight_fraction_leaf", type=float, default=None)
    parser.add_argument("--max_features", type=float, default=None)
    parser.add_argument("--max_leaf_nodes", type=int, default=None, help="Use -1 for None.")
    parser.add_argument("--min_impurity_decrease", type=float, default=None)
    parser.add_argument("--bootstrap", action="store_true", default=None)
    parser.add_argument("--oob_score", action="store_true", default=None)
    parser.add_argument("--verbose", type=int, default=None)
    parser.add_argument("--warm_start", action="store_true", default=None)
    parser.add_argument("--ccp_alpha", type=float, default=None)
    parser.add_argument("--max_samples", type=float, default=None)
    args = parser.parse_args()

    configure_cpu_environment(args.model_n_jobs)
    set_seed(args.seed)
    feature_dir = resolve_feature_dir(args)
    out_dir = default_out_dir(args, feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_path = out_dir / "final_results_test.csv"
    model_path = out_dir / "model.joblib"
    if done_path.exists() and model_path.exists() and not args.overwrite:
        print(f"Skipping completed run: {out_dir}", flush=True)
        return

    x_train, y_train, meta_train = load_feature_split(feature_dir, "train")
    x_val, y_val, meta_val = load_feature_split(feature_dir, "val")
    x_test, y_test, meta_test = load_feature_split(feature_dir, "test")
    if len(y_train) == 0:
        raise ValueError(f"Training split is empty after filtering: {feature_dir}")

    from sklearn.ensemble import ExtraTreesRegressor

    random_state = args.seed if args.random_state_mode == "seed" else None
    params = original_extra_trees_params(
        model_n_jobs=args.model_n_jobs,
        random_state=random_state,
        overrides=build_tree_overrides(args),
    )
    started = time.time()
    model = ExtraTreesRegressor(**params)
    with maybe_threadpool_limits(args.model_n_jobs):
        model.fit(x_train, y_train)

    joblib.dump(model, model_path)
    atomic_save_json(
        out_dir / "run_config.json",
        {
            "feature_dir": str(feature_dir),
            "out_dir": str(out_dir),
            "seed": args.seed,
            "random_state_mode": args.random_state_mode,
            "extra_trees_params": params,
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
        },
    )

    summary = {
        "elapsed_seconds": time.time() - started,
        "feature_dir": str(feature_dir),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "model_n_jobs": args.model_n_jobs,
    }
    for split_name, x, y, meta in (
        ("train", x_train, y_train, meta_train),
        ("val", x_val, y_val, meta_val),
        ("test", x_test, y_test, meta_test),
    ):
        pred = model.predict(x)
        metrics = regression_metrics(y, pred)
        metrics.update(
            {
                "split": split_name,
                "rows": int(len(y)),
                "seed": args.seed,
                "model_n_jobs": args.model_n_jobs,
            }
        )
        atomic_save_csv(out_dir / f"final_results_{split_name}.csv", pd.DataFrame([metrics]))
        save_predictions(out_dir / f"predictions_{split_name}.csv", meta, y, pred)
        for key, value in metrics.items():
            if key not in {"split"}:
                summary[f"{split_name}_{key}"] = value
    atomic_save_json(out_dir / "run_summary.json", summary)
    atomic_save_csv(out_dir / "run_summary.csv", pd.DataFrame([summary]))
    print(f"Saved run to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
