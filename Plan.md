# UniKP EMULaToR Bench Plan

## Summary

This bench adds EMULaToR split retraining wrappers under `emulator_bench/` while preserving UniKP's original modeling path:

- protein representation from frozen `Rostlab/prot_t5_xl_uniref50`
- substrate representation from the repository SMILES Transformer weights
- concatenated 2048-dimensional feature vectors
- sklearn `ExtraTreesRegressor` as the predictor

The original UniKP scripts are not modified.

## Key Behaviors

- Dataset root defaults to `/home/adhil/github/EMULaToR/data/processed/baselines/UniKP`.
- Split files are discovered from grouped random split folders and threshold split folders.
- Default input columns are `sequence`, `smiles`, and `log10_value`.
- Protein and SMILES embeddings are cached once under the dataset root and reused by all retraining, Optuna, and multirun commands.
- Original ExtraTrees settings are encoded explicitly from the original code's bare `ExtraTreesRegressor()` usage; only CPU parallelism (`n_jobs`) and reproducible `random_state` are exposed as bench controls.

## Implementation Files

- `emulator_bench/common.py`: shared paths, split discovery, metrics, atomic writes, and original ExtraTrees parameter defaults.
- `emulator_bench/feature_pipeline.py`: UniKP-compatible ProtT5 and SMILES Transformer embedding code with AMP for embedding inference.
- `emulator_bench/cache_embeddings.py`: one-time cache builder for unique protein sequences and SMILES strings.
- `emulator_bench/build_split_features.py`: materializes train/val/test feature matrices from cached embeddings.
- `emulator_bench/train_single_target_tvt.py`: trains and evaluates one explicit train/val/test job.
- `emulator_bench/launch_parallel_retrain_original.py`: primary multirun entrypoint for original-setting retraining with CPU controls.
- `emulator_bench/tune_optuna.py`: optional later-use Optuna tuner over ExtraTrees hyperparameters.
- `emulator_bench/launch_parallel_retrain_from_optuna.py`: optional later-use multirun retraining from Optuna best parameters.

## Test Plan

- Run all bench commands with `conda run -n mldb`.
- Smoke test with `CUDA_VISIBLE_DEVICES=1` and small row caps.
- Verify cache reuse, completed-run skipping, CPU control flags, and Optuna import/trial wiring.

## Assumptions

- `n_jobs` controls CPU parallelism and does not change model validity.
- The original code's bare `ExtraTreesRegressor()` calls mean sklearn defaults are the original predictor settings.
- `random_state=seed` is the bench default for reproducible multiruns; `--random_state_mode none` restores the original constructor behavior.
