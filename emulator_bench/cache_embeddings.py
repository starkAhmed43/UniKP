import argparse
import time
from pathlib import Path

import pandas as pd
import torch

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    atomic_save_json,
    discover_split_jobs,
    normalize_sequence,
    normalize_smiles,
    normalize_threshold_args,
    protein_cache_path,
    read_table,
    require_columns,
    smiles_cache_path,
)
from emulator_bench.feature_pipeline import (
    PROT_MODEL_ID,
    embed_protein_sequences,
    embed_smiles_values,
    save_npz_atomic,
)


def prepare_frame(frame: pd.DataFrame, args, path: Path) -> pd.DataFrame:
    require_columns(frame, [args.sequence_col, args.smiles_col], path)
    if args.max_rows_per_split is not None:
        frame = frame.head(int(args.max_rows_per_split)).copy()
    if args.drop_dot_smiles:
        frame = frame[~frame[args.smiles_col].astype(str).str.contains(".", regex=False)].copy()
    return frame


def collect_unique_values(jobs, args):
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            path = Path(job[split_key])
            frame = prepare_frame(read_table(path), args, path)
            sequences.update(normalize_sequence(value) for value in frame[args.sequence_col].astype(str))
            smiles_values.update(normalize_smiles(value) for value in frame[args.smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def cache_proteins(args, sequences):
    pending = [seq for seq in sequences if args.overwrite or not protein_cache_path(args.embeddings_dir, seq).exists()]
    if not pending:
        print("Protein cache is already complete.", flush=True)
        return {"total": len(sequences), "written": 0}

    device = torch.device(args.device)
    embedded = embed_protein_sequences(
        pending,
        device=device,
        batch_size=args.protein_batch_size,
        max_tokens=args.protein_max_tokens,
    )
    written = 0
    for sequence, vector in embedded.items():
        save_npz_atomic(
            protein_cache_path(args.embeddings_dir, sequence),
            {
                "vector": vector,
                "length": pd.Series([len(sequence)], dtype="int32").to_numpy(),
            },
        )
        written += 1
    return {"total": len(sequences), "written": written}


def cache_smiles(args, smiles_values):
    pending = [smiles for smiles in smiles_values if args.overwrite or not smiles_cache_path(args.embeddings_dir, smiles).exists()]
    if not pending:
        print("SMILES cache is already complete.", flush=True)
        return {"total": len(smiles_values), "written": 0}

    device = torch.device(args.device)
    embedded = embed_smiles_values(pending, device=device, batch_size=args.smiles_batch_size)
    written = 0
    for smiles, vector in embedded.items():
        save_npz_atomic(
            smiles_cache_path(args.embeddings_dir, smiles),
            {
                "vector": vector,
                "length": pd.Series([len(smiles)], dtype="int32").to_numpy(),
            },
        )
        written += 1
    return {"total": len(smiles_values), "written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache UniKP ProtT5 and SMILES Transformer embeddings for EMULaToR splits.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protein_batch_size", type=int, default=4)
    parser.add_argument("--protein_max_tokens", type=int, default=4096)
    parser.add_argument("--smiles_batch_size", type=int, default=512)
    parser.add_argument("--max_rows_per_split", type=int, default=None)
    parser.add_argument("--keep_dot_smiles", action="store_true", help="Keep multi-component SMILES. Default drops them to match UniKP kcat training.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.drop_dot_smiles = not args.keep_dot_smiles
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    started = time.time()
    sequences, smiles_values = collect_unique_values(jobs, args)
    print(f"Discovered {len(jobs)} split jobs", flush=True)
    print(f"Unique protein sequences: {len(sequences)}", flush=True)
    print(f"Unique SMILES strings: {len(smiles_values)}", flush=True)

    protein_stats = cache_proteins(args, sequences)
    smiles_stats = cache_smiles(args, smiles_values)
    manifest = {
        "cache_version": 1,
        "base_dir": str(args.base_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "protein_model": PROT_MODEL_ID,
        "smiles_model": "UniKP repository trfm_12_23000.pkl",
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "split_groups": list(args.split_groups),
        "thresholds": args.thresholds,
        "drop_dot_smiles": bool(args.drop_dot_smiles),
        "max_rows_per_split": args.max_rows_per_split,
        "protein_cache": protein_stats,
        "smiles_cache": smiles_stats,
        "elapsed_seconds": time.time() - started,
    }
    atomic_save_json(args.embeddings_dir / "manifest.json", manifest)
    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
