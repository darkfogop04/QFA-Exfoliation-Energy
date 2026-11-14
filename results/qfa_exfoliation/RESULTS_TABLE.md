# QFA-Net on exfoliation datasets

Same split for every model (5 random folds, seed 42); validation = 10 % of the training part; test fold used once. Std is over folds (population).

### JARVIS-DFT 3D (bulk): test MAE, meV/atom (lower is better)

fold 0: 163 test, 60 min training; fold 1: 163 test, 42 min training; fold 2: 163 test, 43 min training; fold 3: 162 test, 33 min training; fold 4: 162 test, 36 min training

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
|---|---|---|---|---|---|---|
| QFA classical | 28.22 | 23.39 | 33.61 | 26.78 | 25.59 | 27.52 ± 3.43 |
| QFA mlp_residual | 28.22 | 23.39 | 33.62 | 26.90 | 25.59 | 27.54 ± 3.43 |
| QFA hybrid | 28.22 | 24.00 | 33.61 | 26.77 | 25.59 | 27.64 ± 3.29 |
| QFA hybrid_pruned | 28.22 | 23.98 | 33.61 | 26.83 | 25.59 | 27.65 ± 3.29 |
| matminer XGBoost (open-source, JARVIS-Leaderboard #1 recipe) | 37.74 | 36.01 | 45.75 | 34.65 | 37.36 | 38.30 ± 3.88 |
| QFA-Net main (QT + pruned) | 45.99 | 29.35 | 43.51 | 38.57 | 34.87 | 38.46 ± 5.97 |
| matminer RF (open-source, JARVIS-Leaderboard #2 recipe) | 37.23 | 37.03 | 46.46 | 35.96 | 38.64 | 39.06 ± 3.80 |
| RF on our descriptors | 42.03 | 46.24 | 50.01 | 45.91 | 41.39 | 45.12 ± 3.14 |
| Median predictor | 60.58 | 42.23 | 59.55 | 53.42 | 50.50 | 53.26 ± 6.67 |
| Dummy (predicts the training mean) | 66.83 | 56.42 | 65.64 | 64.31 | 60.77 | 62.79 ± 3.78 |

### JARVIS-DFT 2D (monolayers): test MAE, meV/atom (lower is better)

fold 0: 150 test, 51 min training; fold 1: 150 test, 36 min training; fold 2: 150 test, 35 min training; fold 3: 149 test, 40 min training; fold 4: 149 test, 35 min training

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
|---|---|---|---|---|---|---|
| QFA hybrid_pruned | 21.90 | 33.81 | 37.47 | 23.07 | 30.02 | 29.26 ± 6.02 |
| QFA mlp_residual | 21.90 | 33.72 | 37.93 | 23.88 | 29.43 | 29.37 ± 5.97 |
| QFA hybrid | 21.90 | 33.83 | 37.47 | 23.88 | 30.04 | 29.43 ± 5.86 |
| QFA classical | 21.90 | 33.83 | 37.99 | 23.88 | 29.88 | 29.50 ± 6.00 |
| QFA-Net main (QT + pruned) | 30.74 | 38.88 | 46.83 | 29.60 | 39.14 | 37.04 ± 6.30 |
| matminer XGBoost (open-source, JARVIS-Leaderboard #1 recipe) | 35.17 | 50.17 | 54.25 | 37.17 | 60.66 | 47.48 ± 9.85 |
| matminer RF (open-source, JARVIS-Leaderboard #2 recipe) | 38.13 | 51.17 | 57.94 | 38.99 | 63.54 | 49.96 ± 10.10 |
| RF on our descriptors | 41.33 | 56.12 | 60.66 | 41.59 | 61.06 | 52.15 ± 8.90 |
| Median predictor | 52.54 | 60.68 | 62.58 | 49.14 | 66.94 | 58.38 ± 6.57 |
| Dummy (predicts the training mean) | 65.93 | 69.58 | 75.99 | 67.33 | 76.56 | 71.07 ± 4.40 |

### 2DMatPedia (monolayers): test MAE, meV/atom (lower is better)

fold 0: 906 test, 64 min training; fold 1: 906 test, 43 min training; fold 2: 905 test, 47 min training; fold 3: 905 test, 43 min training; fold 4: 905 test, 45 min training

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
|---|---|---|---|---|---|---|
| QFA hybrid_pruned | 47.03 | 38.01 | 40.41 | 37.76 | 38.26 | 40.29 ± 3.50 |
| QFA mlp_residual | 47.05 | 38.27 | 40.41 | 37.74 | 38.35 | 40.37 ± 3.47 |
| QFA hybrid | 47.71 | 38.18 | 40.41 | 37.74 | 38.25 | 40.46 ± 3.74 |
| QFA classical | 47.68 | 37.98 | 40.41 | 38.09 | 38.96 | 40.62 ± 3.64 |
| QFA-Net main (QT + pruned) | 71.33 | 52.33 | 54.43 | 53.11 | 54.32 | 57.10 ± 7.16 |
| matminer RF (open-source, JARVIS-Leaderboard #2 recipe) | 79.13 | 66.00 | 64.79 | 58.81 | 65.81 | 66.91 ± 6.65 |
| matminer XGBoost (open-source, JARVIS-Leaderboard #1 recipe) | 76.70 | 67.18 | 64.67 | 58.34 | 68.62 | 67.10 ± 5.95 |
| RF on our descriptors | 83.46 | 71.43 | 69.70 | 65.78 | 73.64 | 72.80 ± 5.92 |
| Median predictor | 107.59 | 107.28 | 99.21 | 99.56 | 96.74 | 102.08 ± 4.48 |
| Dummy (predicts the training mean) | 124.76 | 121.06 | 116.56 | 118.01 | 117.39 | 119.56 ± 3.01 |
