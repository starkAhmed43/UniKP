# UniKP emulator_bench

This directory adds EMULaToR split retraining wrappers for UniKP without modifying the original repository scripts.

## Model Inputs

Each sample uses:

- `sequence`: enzyme amino acid sequence
- `smiles`: substrate SMILES
- `log10_value`: default regression target

The feature vector is the original UniKP representation:

- 1024-dimensional protein vector from frozen `Rostlab/prot_t5_xl_uniref50`
- 1024-dimensional substrate vector from the repository SMILES Transformer (`vocab.pkl`, `trfm_12_23000.pkl`)
- concatenated 2048-dimensional input to sklearn `ExtraTreesRegressor`

The default preprocessing drops SMILES containing `.` to match the original UniKP kcat training script. Pass `--keep_dot_smiles` to keep multi-component SMILES.

## Cache Layout

Default dataset root:

```bash
/home/adhil/github/EMULaToR/data/processed/baselines/UniKP
```

Embedding caches are written once under:

```bash
<base_dir>/embeddings/proteins/<hash-prefix>/<hash>.npz
<base_dir>/embeddings/smiles_trfm/<hash-prefix>/<hash>.npz
<base_dir>/embeddings/manifest.json
```

Split feature matrices are materialized under:

```bash
<base_dir>/feature_matrices/<split_group>/<split_name>/{train,val,test}.npz
<base_dir>/feature_matrices/<split_group>/<split_name>/{train,val,test}_metadata.csv
```

Repeated cache, feature, retrain, and Optuna commands skip completed artifacts unless an overwrite flag is passed.

## Original ExtraTrees Settings

The original UniKP code uses `ExtraTreesRegressor()` with no explicit hyperparameters. The bench encodes those defaults explicitly:

- `n_estimators=100`
- `criterion="squared_error"`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `max_features=1.0`
- `bootstrap=False`
- `warm_start=False`

`--model_n_jobs` is the CPU-control exception. It sets sklearn `n_jobs` and related thread environment variables. The default `--random_state_mode seed` makes multiruns reproducible; `--random_state_mode none` restores the original constructor behavior.

## Primary Workflow

Run from the repository root with the `mldb` conda environment. To use physical GPU 1 for embedding, mask it and use `cuda:0` inside the process:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n mldb python emulator_bench/launch_parallel_retrain_original.py \
  --cache_device cuda:0 \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles enzyme_sequence_splits substrate_splits conformer_cosine_splits enzyme_structure_splits uniprot_time_splits \
  --seeds 0 1 2 3 4 \
  --num_workers 4 \
  --model_n_jobs 8
```

This command builds missing embedding caches, builds missing feature matrices, and launches original-setting retraining jobs in parallel. If interrupted, rerun the same command; completed caches, matrices, and runs are skipped.

## Optuna

Optuna is included for later use and tunes only ExtraTrees hyperparameters. Build caches and matrices first, then run:

```bash
conda run -n mldb python emulator_bench/tune_optuna.py \
  --split_groups random_splits_grouped_sequence \
  --seeds 0 \
  --model_n_jobs 8 \
  --n_trials 20 \
  --study_name unikp_optuna_random_sequence
```

Retrain from the best study result:

```bash
conda run -n mldb python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --hparams_json /home/adhil/github/EMULaToR/data/processed/baselines/UniKP/optuna_studies/unikp_optuna_random_sequence_best_hparams.json \
  --split_groups random_splits_grouped_sequence \
  --seeds 0 1 2 3 4 \
  --num_workers 4 \
  --model_n_jobs 8
```
