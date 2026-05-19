import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import numpy as np
import optuna
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_FEATURE_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    atomic_save_csv,
    configure_cpu_environment,
    discover_split_jobs,
    feature_job_dir,
    metric_direction,
    normalize_threshold_args,
    original_extra_trees_params,
    regression_metrics,
    set_seed,
)


def sqlite_path_from_storage(storage):
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    raw_path = unquote(parsed.path or "")
    return Path(raw_path) if raw_path else None


def sqlite_has_optuna_schema(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return "version_info" in tables


def prepare_storage(args):
    db_path = sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return
    if args.reset_storage:
        db_path.unlink()
        return
    if not sqlite_has_optuna_schema(db_path):
        raise RuntimeError(f"Existing sqlite file is not an Optuna database: {db_path}")


def load_split(feature_dir: Path, split: str):
    with np.load(Path(feature_dir) / f"{split}.npz", allow_pickle=False) as data:
        return data["X"].astype(np.float32, copy=False), data["y"].astype(np.float32, copy=False)


def suggest_hparams(trial):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500, 800]),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
    }
    if params["bootstrap"]:
        params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
    else:
        params["max_samples"] = None
    max_depth_choice = trial.suggest_categorical("max_depth", [-1, 10, 20, 40, 80])
    params["max_depth"] = None if max_depth_choice == -1 else max_depth_choice
    return params


def score_one(feature_dir: Path, seed: int, hparams: dict, args):
    from sklearn.ensemble import ExtraTreesRegressor

    x_train, y_train = load_split(feature_dir, "train")
    x_eval, y_eval = load_split(feature_dir, args.eval_split)
    random_state = seed if args.random_state_mode == "seed" else None
    params = original_extra_trees_params(
        model_n_jobs=args.model_n_jobs,
        random_state=random_state,
        overrides=hparams,
    )
    model = ExtraTreesRegressor(**params)
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    metrics = regression_metrics(y_eval, pred)
    return float(metrics[args.metric]), metrics


def main():
    parser = argparse.ArgumentParser(description="Optional Optuna tuning for UniKP ExtraTrees hyperparameters.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--feature_dirname", type=str, default=DEFAULT_FEATURE_DIRNAME)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--model_n_jobs", type=int, default=None)
    parser.add_argument("--random_state_mode", choices=["seed", "none"], default="seed")
    parser.add_argument("--metric", choices=["rmse", "mse", "mae", "r2_score", "pearson", "spearman"], default="rmse")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="unikp_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    configure_cpu_environment(args.model_n_jobs)
    set_seed(args.sampler_seed)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = f"sqlite:///{Path(args.base_dir) / 'optuna_studies' / (args.study_name + '.db')}"
    prepare_storage(args)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    feature_dirs = [
        feature_job_dir(Path(args.base_dir), args.feature_dirname, job["split_group"], job["split_name"])
        for job in jobs
    ]

    study = optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    def objective(trial):
        hparams = suggest_hparams(trial)
        scores = []
        for feature_dir in feature_dirs:
            for seed in args.seeds:
                score, _metrics = score_one(feature_dir, seed, hparams, args)
                scores.append(score)
        trial.set_user_attr("n_scores", len(scores))
        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=args.n_trials)

    out_dir = Path(args.base_dir) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = out_dir / f"{args.study_name}_trials.csv"
    best_json = out_dir / f"{args.study_name}_best_hparams.json"
    atomic_save_csv(trials_csv, study.trials_dataframe())
    with open(best_json, "w") as handle:
        json.dump(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "direction": study.direction.name.lower(),
                "metric": args.metric,
                "eval_split": args.eval_split,
                "best_trial_number": int(study.best_trial.number),
                "best_value": float(study.best_value),
                "best_hparams": dict(study.best_params),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved best hparams to {best_json}", flush=True)


if __name__ == "__main__":
    main()
