import csv
import hashlib
import inspect
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "UniKP"
DEFAULT_BASE_DIR = Path(f"/home/adhil/github/EMULaToR/data/processed/baselines/{DEFAULT_MODEL_NAME}")
DEFAULT_EMBEDDINGS_DIR = DEFAULT_BASE_DIR / "embeddings"
DEFAULT_FEATURE_DIRNAME = "feature_matrices"
DEFAULT_RESULTS_DIRNAME = "unikp_original_retrain"
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "enzyme_sequence_splits",
    "substrate_splits",
    "conformer_cosine_splits",
    "enzyme_structure_splits",
    "uniprot_time_splits",
    "group_shuffle_splits",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split())


def normalize_smiles(smiles: str) -> str:
    return str(smiles).strip()


def protein_cache_key(sequence: str) -> str:
    return stable_hash(normalize_sequence(sequence))


def smiles_cache_key(smiles: str) -> str:
    return stable_hash(normalize_smiles(smiles))


def protein_cache_path(embeddings_dir: Path, sequence: str) -> Path:
    key = protein_cache_key(sequence)
    return Path(embeddings_dir) / "proteins" / key[:2] / f"{key}.npz"


def smiles_cache_path(embeddings_dir: Path, smiles: str) -> Path:
    key = smiles_cache_key(smiles)
    return Path(embeddings_dir) / "smiles_trfm" / key[:2] / f"{key}.npz"


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def atomic_save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = Path(path).exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _is_direct_split_group(split_group: str) -> bool:
    return split_group.startswith("random_splits_grouped_") or split_group in {"uniprot_time_splits", "group_shuffle_splits"}


def _threshold_value(name: str) -> float:
    try:
        return float(name.split("threshold_")[-1])
    except Exception:
        return math.inf


def _difficulty_labels_for_thresholds(names: List[str]) -> Dict[str, str]:
    ordered = sorted(names, key=_threshold_value)
    if len(ordered) == 1:
        return {ordered[0]: "single"}
    if len(ordered) == 2:
        return {ordered[0]: "hard", ordered[1]: "easy"}
    if len(ordered) == 3:
        return {ordered[0]: "hard", ordered[1]: "medium", ordered[2]: "easy"}
    return {name: f"rank_{idx}" for idx, name in enumerate(ordered, start=1)}


def normalize_threshold_args(thresholds: Optional[Iterable[str]], threshold: Optional[str] = None) -> Optional[List[str]]:
    values: List[str] = []
    if thresholds:
        values.extend(str(value) for value in thresholds if str(value).strip())
    if threshold and str(threshold).strip():
        values.append(str(threshold))
    if not values:
        return None
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    base_dir = Path(base_dir)
    split_groups = list(split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = set(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []

    for split_group in split_groups:
        group_dir = base_dir / split_group
        if not group_dir.exists():
            continue

        train_path = _find_split_file(group_dir, "train")
        val_path = _find_split_file(group_dir, "val")
        test_path = _find_split_file(group_dir, "test")
        if train_path and val_path and test_path:
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": split_group if _is_direct_split_group(split_group) else group_dir.name,
                    "difficulty": split_group,
                    "root_dir": str(group_dir),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
            continue

        candidate_dirs = []
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            if threshold_filter is not None and child.name not in threshold_filter:
                continue
            if child.name.startswith("threshold_") or child.name in {"easy", "medium", "hard"}:
                candidate_dirs.append(child)

        threshold_names = [child.name for child in candidate_dirs if child.name.startswith("threshold_")]
        difficulty_by_name = _difficulty_labels_for_thresholds(threshold_names)
        for child in candidate_dirs:
            train_path = _find_split_file(child, "train")
            val_path = _find_split_file(child, "val")
            test_path = _find_split_file(child, "test")
            if not (train_path and val_path and test_path):
                continue
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "difficulty": difficulty_by_name.get(child.name, child.name),
                    "root_dir": str(child),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
    return jobs


def resolve_single_split_job(base_dir: Path, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    thresholds = None if _is_direct_split_group(split_group) else normalize_threshold_args(None, threshold)
    jobs = discover_split_jobs(base_dir, split_groups=[split_group], thresholds=thresholds)
    if not jobs:
        detail = f"{split_group}/{threshold}" if threshold else split_group
        raise FileNotFoundError(f"No split job found for {detail} in {base_dir}")
    if threshold is None or _is_direct_split_group(split_group):
        if len(jobs) > 1 and not _is_direct_split_group(split_group):
            available = ", ".join(job["split_name"] for job in jobs)
            raise ValueError(f"Multiple jobs found for {split_group}; specify --threshold. Available: {available}")
        return jobs[0]
    for job in jobs:
        if job["split_name"] == threshold:
            return job
    available = ", ".join(job["split_name"] for job in jobs)
    raise FileNotFoundError(f"Threshold {threshold} not found for {split_group}. Available: {available}")


def feature_job_dir(base_dir: Path, feature_dirname: str, split_group: str, split_name: str) -> Path:
    safe_group = str(split_group).replace("/", "_")
    safe_name = str(split_name).replace("/", "_")
    return Path(base_dir) / feature_dirname / safe_group / safe_name


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2_score": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    if y_true.size < 2 or np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2_score": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "pearson": pearson,
        "spearman": spearman,
    }


def metric_direction(metric: str) -> str:
    return "minimize" if metric in {"mse", "rmse", "mae"} else "maximize"


def original_extra_trees_params(model_n_jobs=None, random_state=None, overrides: Optional[Dict] = None) -> Dict:
    params = {
        "n_estimators": 100,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": False,
        "oob_score": False,
        "n_jobs": model_n_jobs,
        "random_state": random_state,
        "verbose": 0,
        "warm_start": False,
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    }
    if overrides:
        params.update({key: value for key, value in overrides.items() if value is not None})

    from sklearn.ensemble import ExtraTreesRegressor

    accepted = set(inspect.signature(ExtraTreesRegressor).parameters)
    return {key: value for key, value in params.items() if key in accepted}


def summarize_seed_runs(rows: List[Dict], group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    present_metrics = []
    for col in metric_cols:
        if col not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            frame[col] = numeric
            present_metrics.append(col)
    if not present_metrics:
        return frame[group_cols].drop_duplicates()
    grouped = frame.groupby(group_cols, dropna=False)[present_metrics]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def configure_cpu_environment(model_n_jobs: Optional[int]) -> None:
    if model_n_jobs is None or int(model_n_jobs) <= 0:
        return
    value = str(int(model_n_jobs))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, value)
