# QFA vs the best open-source models, per exfoliation dataset (13 Nov 2026)

Test MAE in meV/atom, lower is better. Every comparison uses the same test materials for all models.

## JARVIS-DFT 3D: official JARVIS-Leaderboard split (650 train + 81 val, 81 test)
Leaderboard rows = published scores (pages.nist.gov/jarvis_leaderboard, dft_3d_exfoliation_energy).
QFA: `experiments/run_jarvis_leaderboard_split.py` -> `jarvis_dft_3d_leaderboard/test.json` (3 seeds, test used once).

| Rank | Model | MAE |
|---|---|---|
| 1 | QFA classical / hybrid / hybrid_pruned / mlp_residual (identical: residual stayed 0) | 30.49 |
| 2 | QFA-Net main (QT + pruned) | 32.01 |
| 3 | matminer_xgboost (UofT) | 40.89 |
| 4 | matminer_rf (UofT) | 42.76 |
| 5 | kgcnn_cgcnn | 45.76 |
| 6 | kgcnn_dimenetPP | 46.15 |
| 7 | kgcnn_coNGN | 46.27 |
| 8 | kgcnn_coGN | 47.70 |
| 9 | kgcnn_schnet | 48.30 |
| 10 | matminer_lgbm | 49.51 |
| 11 | matformer | 51.66 |
| 12 | alignn_model | 52.70 |
| 13 | cgcnn_model | 52.70 |
| 14 | cfid | 62.12 |
| 15 | cfid_chem | 63.77 |
| 16 | kgcnn_megnet | 68.24 |

Our re-runs of the two top recipes on this split: matminer_xgboost 42.71 (published 40.89), matminer_rf 44.33
(published 42.76). Median predictor 48.84, Dummy 60.28.

## Our 5 random folds (all three datasets): top two JARVIS-Leaderboard recipes re-run on the same folds
`experiments/run_matminer_rf.py --model rf|xgb`; full per-fold tables in `RESULTS_TABLE.md`.

| Dataset | QFA best (5-fold mean) | QFA-Net main (QT) | matminer XGBoost (#1) | matminer RF (#2) | Dummy | QFA best beats XGBoost on |
|---|---|---|---|---|---|---|
| JARVIS 3D (813) | 27.52 (classical) | 38.46 | 38.30 | 39.06 | 62.79 | 5/5 folds |
| JARVIS 2D (748) | 29.26 (hybrid_pruned) | 37.04 | 47.48 | 49.96 | 71.07 | 5/5 folds |
| 2DMatPedia (4,527) | 40.29 (hybrid_pruned) | 57.10 | 67.10 | 66.91 | 119.56 | 5/5 folds |

QFA-Net main (QT) vs XGBoost: 3/5 folds on JARVIS 3D (means 38.46 vs 38.30, a tie), 5/5 on JARVIS 2D and 2DMatPedia.
On the monolayer sets 127 of the 273 matminer structure features fail for 74-92 % of materials (vacuum), as in the
original recipe; they are imputed.

## MatBench jdft2d (subset of JARVIS 3D, official 5 folds): see `results/qfa/RESULTS_TABLE.md`
QFA variants 27.16-27.27, MODNet v0.1.12 33.19 (#1 on MatBench), QFA-Net main (QT) 35.30 (rank 3 of 9).
