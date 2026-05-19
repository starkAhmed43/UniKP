import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_FEATURE_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    atomic_save_csv,
    atomic_save_json,
    discover_split_jobs,
    feature_job_dir,
    normalize_sequence,
    normalize_smiles,
    normalize_threshold_args,
    protein_cache_path,
    read_table,
    require_columns,
    smiles_cache_path,
)
from emulator_bench.feature_pipeline import load_cached_vector, save_npz_atomic


def prepare_frame(frame: pd.DataFrame, args, path: Path) -> pd.DataFrame:
    require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], path)
    if args.max_rows_per_split is not None:
        frame = frame.head(int(args.max_rows_per_split)).copy()
    if args.drop_dot_smiles:
        frame = frame[~frame[args.smiles_col].astype(str).str.contains(".", regex=False)].copy()
    frame = frame[pd.notna(frame[args.target_col])].copy()
    return frame.reset_index(drop=True)


def build_one_split(frame: pd.DataFrame, args, out_dir: Path, split_name: str, source_path: Path) -> dict:
    matrix_path = out_dir / f"{split_name}.npz"
    meta_path = out_dir / f"{split_name}_metadata.csv"
    if matrix_path.exists() and meta_path.exists() and not args.overwrite:
        return {"split": split_name, "status": "skipped_exists", "rows": int(len(frame)), "matrix_path": str(matrix_path)}

    features = []
    y = []
    kept_rows = []
    for row_index, row in frame.iterrows():
        sequence = normalize_sequence(row[args.sequence_col])
        smiles = normalize_smiles(row[args.smiles_col])
        protein_path = protein_cache_path(args.embeddings_dir, sequence)
        smiles_path = smiles_cache_path(args.embeddings_dir, smiles)
        if not protein_path.exists():
            raise FileNotFoundError(f"Missing protein cache for row {row_index}: {protein_path}")
        if not smiles_path.exists():
            raise FileNotFoundError(f"Missing SMILES cache for row {row_index}: {smiles_path}")
        smiles_vec = load_cached_vector(smiles_path)
        protein_vec = load_cached_vector(protein_path)
        if smiles_vec.shape[0] != 1024 or protein_vec.shape[0] != 1024:
            raise ValueError(f"Expected 1024+1024 features, got {smiles_vec.shape[0]}+{protein_vec.shape[0]}")
        features.append(np.concatenate([smiles_vec, protein_vec]).astype(np.float32, copy=False))
        y.append(float(row[args.target_col]))
        kept_rows.append(row.to_dict())

    x_array = np.vstack(features).astype(np.float32, copy=False) if features else np.zeros((0, 2048), dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    save_npz_atomic(matrix_path, {"X": x_array, "y": y_array})
    meta = pd.DataFrame(kept_rows)
    meta.insert(0, "source_row", range(len(meta)))
    meta.insert(0, "source_path", str(source_path))
    atomic_save_csv(meta_path, meta)
    return {"split": split_name, "status": "written", "rows": int(len(frame)), "matrix_path": str(matrix_path)}


def main():
    parser = argparse.ArgumentParser(description="Materialize UniKP train/val/test feature matrices from cached embeddings.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--feature_dirname", type=str, default=DEFAULT_FEATURE_DIRNAME)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--keep_dot_smiles", action="store_true", help="Keep multi-component SMILES. Default drops them to match UniKP kcat training.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.drop_dot_smiles = not args.keep_dot_smiles

    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    started = time.time()
    summary_rows = []
    for job in jobs:
        out_dir = feature_job_dir(args.base_dir, args.feature_dirname, job["split_group"], job["split_name"])
        out_dir.mkdir(parents=True, exist_ok=True)
        job_rows = []
        for split_name in ("train", "val", "test"):
            path = Path(job[f"{split_name}_path"])
            frame = prepare_frame(read_table(path), args, path)
            job_rows.append(build_one_split(frame, args, out_dir, split_name, path))
        manifest = {
            "feature_version": 1,
            "split_group": job["split_group"],
            "split_name": job["split_name"],
            "difficulty": job["difficulty"],
            "root_dir": job["root_dir"],
            "embeddings_dir": str(args.embeddings_dir),
            "sequence_col": args.sequence_col,
            "smiles_col": args.smiles_col,
            "target_col": args.target_col,
            "drop_dot_smiles": bool(args.drop_dot_smiles),
            "max_rows_per_split": args.max_rows_per_split,
            "splits": job_rows,
        }
        atomic_save_json(out_dir / "manifest.json", manifest)
        for row in job_rows:
            row.update({"split_group": job["split_group"], "split_name": job["split_name"], "difficulty": job["difficulty"]})
            summary_rows.append(row)

    summary_path = args.base_dir / args.feature_dirname / "feature_matrix_index.csv"
    atomic_save_csv(summary_path, pd.DataFrame(summary_rows))
    print(f"Saved feature matrix index to {summary_path}", flush=True)
    print(f"Elapsed seconds: {time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
