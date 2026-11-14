# QFA-Net on exfoliation datasets

Same split for every model (5 random folds, seed 42); validation = 10 % of the training part; test fold used once. Std is over folds (population).

### JARVIS-DFT 3D (bulk): test MAE, meV/atom (lower is better)

fold 0: 163 test, 60 min training; fold 1: 163 test, 42 min training; fold 2: 163 test, 43 min training

| Model | Fold 0 | Fold 1 | Fold 2 | Mean ± std |
|---|---|---|---|---|
| QFA classical | 28.22 | 23.39 | 33.61 | 28.41 ± 4.18 |
| QFA mlp_residual | 28.22 | 23.39 | 33.62 | 28.41 ± 4.18 |
| QFA hybrid_pruned | 28.22 | 23.98 | 33.61 | 28.61 ± 3.94 |
| QFA hybrid | 28.22 | 24.00 | 33.61 | 28.61 ± 3.93 |
| QFA-Net main (QT + pruned) | 45.99 | 29.35 | 43.51 | 39.62 ± 7.33 |
| RF on our descriptors | 42.03 | 46.24 | 50.01 | 46.09 ± 3.26 |
| Median predictor | 60.58 | 42.23 | 59.55 | 54.12 ± 8.42 |
| Dummy (predicts the training mean) | 66.83 | 56.42 | 65.64 | 62.96 ± 4.65 |

### JARVIS-DFT 2D (monolayers): test MAE, meV/atom (lower is better)

fold 0: 150 test, 51 min training; fold 1: 150 test, 36 min training; fold 2: 150 test, 35 min training

| Model | Fold 0 | Fold 1 | Fold 2 | Mean ± std |
|---|---|---|---|---|
| QFA hybrid_pruned | 21.90 | 33.81 | 37.47 | 31.06 ± 6.65 |
| QFA hybrid | 21.90 | 33.83 | 37.47 | 31.07 ± 6.65 |
| QFA mlp_residual | 21.90 | 33.72 | 37.93 | 31.18 ± 6.78 |
| QFA classical | 21.90 | 33.83 | 37.99 | 31.24 ± 6.82 |
| QFA-Net main (QT + pruned) | 30.74 | 38.88 | 46.83 | 38.81 ± 6.57 |
| RF on our descriptors | 41.33 | 56.12 | 60.66 | 52.70 ± 8.25 |
| Median predictor | 52.54 | 60.68 | 62.58 | 58.60 ± 4.35 |
| Dummy (predicts the training mean) | 65.93 | 69.58 | 75.99 | 70.50 ± 4.16 |

### 2DMatPedia (monolayers): test MAE, meV/atom (lower is better)

fold 0: 906 test, 64 min training; fold 1: 906 test, 43 min training; fold 2: 905 test, 47 min training

| Model | Fold 0 | Fold 1 | Fold 2 | Mean ± std |
|---|---|---|---|---|
| QFA hybrid_pruned | 47.03 | 38.01 | 40.41 | 41.82 ± 3.82 |
| QFA mlp_residual | 47.05 | 38.27 | 40.41 | 41.91 ± 3.74 |
| QFA classical | 47.68 | 37.98 | 40.41 | 42.03 ± 4.12 |
| QFA hybrid | 47.71 | 38.18 | 40.41 | 42.10 ± 4.07 |
| QFA-Net main (QT + pruned) | 71.33 | 52.33 | 54.43 | 59.36 ± 8.51 |
| RF on our descriptors | 83.46 | 71.43 | 69.70 | 74.86 ± 6.12 |
| Median predictor | 107.59 | 107.28 | 99.21 | 104.69 ± 3.88 |
| Dummy (predicts the training mean) | 124.76 | 121.06 | 116.56 | 120.79 ± 3.35 |
