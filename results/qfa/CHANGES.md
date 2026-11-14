# Changes made for the final run (10 Nov 2026, laptop)

No git in this folder, so every change is listed here.

## Environment (no code changes)
- `D:\MTP\venv_qmat` no longer exists (no D: drive). New venv: `C:\venvs\qfa` (Python 3.13.1, outside OneDrive).
- Installed: torch 2.15.0+cpu, orb-models 0.7.0, ase 3.29.0, pymatgen 2026.10.24, matminer 0.10.1,
  scikit-learn 1.9.1, numpy 2.5.4, matbench 0.6 (`--no-deps`).
- orb-models 0.7.0 pins `dm-tree==0.1.8`, which has no Python 3.13 wheel (it needs CMake to build).
  Installed `dm-tree 0.1.10` (prebuilt wheel, same API) and `orb-models --no-deps` plus its other
  dependencies (cached-path, scipy, tqdm, nvalchemi-toolkit-ops[torch]).
- Harmless warning at start-up: "Warp: insufficient CUDA driver version" (everything runs on CPU).

## Checks
- `pytest tests/test_qfa_lab.py tests/test_qfa.py -q` -> 14 passed.
- `experiments/check_qfa_setup.py` -> all 6 lines [OK].
- `run_qfa.py --task jdft2d --folds 0 --quick` -> all 6 variants, `qt_materialised_equal` true.

## Code changes
- `experiments/run_qfa.py`, `summary()`: `f[5:]` -> `f[4:]`. For "fold0.json", `f[5:]` is ".json", so no
  fold file was ever found and `--summary` printed "no results yet" for every task. Training, results files
  and `--export` were not affected (export uses its own file filter). Fixed after the run finished.

## Exfoliation datasets (13 Nov 2026)
- Installed `jarvis-tools 2026.6.12` and `requests` in `C:\venvs\qfa`.
- `experiments/download_exfoliation_data.py`, `get_2dmatpedia`: skip links containing "band". The page also links
  the 3.9 GB band-structure tarball (no exfoliation labels); the old code downloaded every link.
- same file, `read_json_any`: choose gzip by the file's magic bytes, not by the ".gz" name. requests had already
  un-gzipped `db.json.gz` (Content-Encoding), so `gzip.open` failed with BadGzipFile.
- New `experiments/convert_exfoliation_extra.py`: converts `data/exfoliation/massive_discovery_2025/raw/databases.zip`
  (Zenodo 10.5281/zenodo.17035156, CC-BY 4.0) to `dataset.json.gz` (2,988 materials, E_exf in meV/A^2, MatterSim).
  The Zenodo zip was found via the paper's data-availability statement and moved from `no_public_data/raw/` into
  its own folder `massive_discovery_2025/`.
- New `experiments/run_exfoliation_all.py`: time-boxed loop over the labelled exfoliation datasets with the same
  seeds and stage caps as run_final.py (tiny: 3 seeds, 8 min; small: 2 seeds, 12 min).
- `convert_exfoliation_extra.py` also converts mc2d_mounet2018 (1,694, `EE_and_PE_structures.txt`) and
  mc2d_campi2022 (2,848, `structure_2d.json` + as-extracted structures; the 9 GB AiiDA file is not needed).
  Findings per folder: `data/exfoliation/DATA_CHECK.md`.
- `run_exfoliation_all.py`: added `--folds` (e.g. fold 0 only); `--dry-run` no longer writes STATUS.md.
- Removed an aborted partial download `twodmatpedia/raw/2dmatpedia_band.json.tar.gz.part`.

## Run settings
- `run_final.py --hours 5` (not 5.5): about 25 min of the 5.5 h budget were used for setup, and the
  run is resumable if time remains. Started 10 Nov 2026 02:39 in a separate PowerShell window;
  log: `results/qfa/run_final_log.txt` (UTF-16). Finished 07:29, no task failed.
- Run 2: `run_final.py --hours 1.75`, 09:15-10:56, log `run_final_log_2.txt` (expt_gap fold 0, glass fold 0;
  glass `--quick` smoke test first, classification path OK). Before it, run-1 files were copied to
  `STATUS_run1.md`, `RESULTS_TABLE_run1.md`, `final_results_run1.json` (run 2 rewrites STATUS.md).
- `RESULTS_TABLE.md` was built from the fold JSONs + `data/matbench_meta/leaderboard_fold_scores.json` by a
  one-off helper script (not part of the repository); `final_results.json` and `SUMMARY.txt` by run_qfa.py.
- Hardware: 13th Gen Intel Core i5-13500H, 12 cores / 16 threads, 15.7 GB RAM, CPU only.
- New `experiments/exfoliation_results_table.py` (13 Nov): per-dataset table of all QFA variants + RF from the
  fold JSONs, plus Median predictor and Dummy (training mean) computed on the same saved `folds.json` split
  (train = everything outside the test fold). Reproduces 2DMatPedia fold 0: median 107.59, dummy 124.76.
  Writes `results/qfa_exfoliation/RESULTS_TABLE.md`. No training code changed.
- Exfoliation run 3 (13 Nov, started 04:43): Windows Update restarted the laptop at 03:29 and killed run 2 in
  jarvis_dft_3d fold 1 (exit 1073807364). Resumed with `run_exfoliation_all.py --hours 8 --folds 0,1,2`,
  log `results/qfa_exfoliation/run_log_3.txt`; run-2 status kept as `STATUS_run2.md`.
- Exfoliation run 3 finished 08:50 (folds 0-2 of all three sets, no failures). Fold 0-2 tables kept as
  `results/qfa_exfoliation/RESULTS_TABLE_folds0-2.md`, status as `STATUS_run3.md`.
- Exfoliation run 4 (13 Nov, started 09:06): folds 3-4, `run_exfoliation_all.py --hours 7 --folds 0,1,2,3,4`,
  log `results/qfa_exfoliation/run_log_4.txt`.
- Exfoliation run 4 finished 12:59 (folds 3-4, no failures). All three sets now have 5 folds;
  `results/qfa_exfoliation/RESULTS_TABLE.md` holds the 5-fold tables.
- Open-source comparison (13 Nov): `data/exfoliation/jarvis_dft_3d/jarvis_leaderboard_split.json` = official
  JARVIS-Leaderboard split of dft_3d exfoliation (650/81/81; all 812 ids and targets equal ours; our extra
  JVASP-30404 is not used). New `experiments/run_jarvis_leaderboard_split.py` (QFA on that split, test once).
  New `experiments/run_matminer_rf.py` = leaderboard model matminer_rf (UofT) re-run with its exact features and
  RF settings; on the leaderboard split it gives 44.33 vs published 42.76 (library versions). Also run on our
  5 folds of all three sets; `exfoliation_results_table.py` adds it as a row. XGBoost (#1, 40.89) added later
  (below).
- The first leaderboard-split QFA run was stopped because the laptop ran low on memory; rerun later.
- Installed xgboost 3.4.1 in C:\venvs\qfa. `run_matminer_rf.py --model xgb` = leaderboard #1
  matminer_xgboost recipe (CPU `hist` instead of `gpu_hist`); leaderboard split 42.71 vs published 40.89.
- QFA on the JARVIS-Leaderboard split (rerun 18:16-19:10): 30.49 (all four direct variants), QT 32.01.
- Summary: `results/qfa_exfoliation/COMPARISON_BEST_MODELS.md`.

## Project clean-up (14 Nov 2026)
- Moved (not deleted) to `../qc-improved_removed_2026-11-14/`: `.github/` (CI workflow that would start a
  120-min GitHub Actions job on every push), a hand-over note, `REPORT.md` (mid-term report with
  unsourced leaderboard numbers), AQUA-Net (never run; `src/aqua`, `run_aqua.py`, `configs/aqua.yaml`,
  `docs/AQUA_NET.md`, `tests/test_aqua*.py`), `experiments/run_all.py` (needed AQUA, never produced results),
  `docs/RUN_FINAL.md` and `docs/QFA_NET.md` (outdated; merged into README.md).
- Deleted only regenerable caches: every `__pycache__/` and `.pytest_cache/`.
- Rewritten `README.md` (full documentation, usage guide, project map and results),
  `requirements.txt` (exact QFA versions), `requirements-legacy.txt` (HQ-GNN), new `.gitignore`
  (keeps results; ignores data/cache and raw downloads). Data folders untouched.
- Tests: `tests/test_qfa.py`, `tests/test_qfa_lab.py` 14 passed. HQ-GNN tests need pennylane/torch-geometric
  (requirements-legacy.txt), not installed in C:\venvs\qfa.
