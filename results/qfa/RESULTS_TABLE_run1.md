# QFA-Net final results (laptop CPU run, 10 Nov 2026)

Scores are test-fold scores on the official MatBench folds. Leaderboard models are scored on the same folds as QFA-Net (their per-fold scores from `data/matbench_meta/leaderboard_fold_scores.json`). Numbers shown with 4 decimals; full precision in `final_results.json`.

## 1. Per-task comparison

### jdft2d (MAE in meV/atom (lower is better)), folds 0, 1, 2, 3, 4

| # | Model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | Mean | Folds |
|---|---|---|---|---|---|---|---|---|
| 1 | QFA control: mlp_residual | 20.7871 | 28.2541 | 40.1528 | 15.2256 | 31.4040 | 27.1647 | 5 |
| 2 | QFA control: hybrid | 20.8092 | 28.2367 | 40.2135 | 15.2256 | 31.4040 | 27.1778 | 5 |
| 3 | QFA control: classical | 20.8948 | 29.0388 | 39.6012 | 15.2256 | 31.4040 | 27.2329 | 5 |
| 4 | QFA control: hybrid_pruned | 20.8356 | 28.6432 | 40.2188 | 15.2256 | 31.4040 | 27.2654 | 5 |
| 5 | modnet_v0.1.12 | 25.5515 | 30.4506 | 45.1925 | 26.9801 | 37.7845 | 33.1918 | 5 |
| 6 | modnet_v0.1.10 | 27.5769 | 27.9722 | 51.3402 | 26.9141 | 38.8806 | 34.5368 | 5 |
| 7 | **QFA-Net (qt_hybrid_pruned)** | 24.3569 | 38.7747 | 52.9537 | 23.2581 | 37.1678 | 35.3022 | 5 |
| 8 | automatminer_expressv2020 | 29.5070 | 44.3036 | 54.4690 | 28.0759 | 42.8931 | 39.8497 | 5 |
| 9 | alignn | 37.0106 | 45.3752 | 58.3624 | 31.9625 | 44.4110 | 43.4244 | 5 |
| 10 | CrabNet | 36.0753 | 45.8800 | 67.1110 | 31.6798 | 47.3058 | 45.6104 | 5 |
| 11 | QFA control: rf_descriptors | 42.5659 | 45.3895 | 63.0650 | 43.2357 | 50.8612 | 49.0234 | 5 |
| 12 | cgcnnv2019 | 34.4937 | 51.1167 | 69.4250 | 42.7453 | 48.4396 | 49.2440 | 5 |
| 13 | rf | 42.7473 | 45.7510 | 66.2421 | 44.0340 | 51.4457 | 50.0440 | 5 |
| 14 | dummy | 53.1447 | 72.8118 | 83.1220 | 61.3174 | 66.0295 | 67.2851 | 5 |

QFA-Net rank: **3 of 9** (leaderboard models + QFA-Net main score; our own controls are listed but not counted).

### steels (MAE in MPa (lower is better)), folds 0, 1, 2

| # | Model | fold 0 | fold 1 | fold 2 | Mean | Folds |
|---|---|---|---|---|---|---|
| 1 | modnet_v0.1.12 | 103.5846 | 72.3160 | 80.5134 | 85.4713 | 3 |
| 2 | RFLR | 97.5404 | 86.2789 | 79.5099 | 87.7764 | 3 |
| 3 | automatminer_expressv2020 | 109.3058 | 80.4188 | 83.5360 | 91.0869 | 3 |
| 4 | modnet_v0.1.10 | 112.2905 | 81.9908 | 99.3739 | 97.8850 | 3 |
| 5 | QFA control: hybrid_pruned | 128.1099 | 71.4847 | 98.8873 | 99.4940 | 3 |
| 6 | QFA control: hybrid | 128.0202 | 78.6946 | 99.1941 | 101.9696 | 3 |
| 7 | rf | 114.6331 | 85.6694 | 110.0055 | 103.4360 | 3 |
| 8 | QFA control: rf_descriptors | 111.7126 | 89.3516 | 110.2116 | 103.7586 | 3 |
| 9 | CrabNet | 116.2240 | 88.0920 | 108.1233 | 104.1464 | 3 |
| 10 | QFA control: classical | 135.5730 | 85.7849 | 106.0789 | 109.1456 | 3 |
| 11 | QFA control: mlp_residual | 135.5730 | 85.7849 | 106.0789 | 109.1456 | 3 |
| 12 | **QFA-Net (qt_hybrid_pruned)** | 124.3287 | 96.1091 | 118.3382 | 112.9253 | 3 |
| 13 | dummy | 241.4591 | 219.3770 | 225.7932 | 228.8764 | 3 |

QFA-Net rank: **7 of 8** (leaderboard models + QFA-Net main score; our own controls are listed but not counted).

### phonons (MAE in cm^-1 (lower is better)), folds 0

| # | Model | fold 0 | Mean | Folds |
|---|---|---|---|---|
| 1 | QFA control: mlp_residual | 27.1799 | 27.1799 | 1 |
| 2 | QFA control: hybrid | 27.4825 | 27.4825 | 1 |
| 3 | QFA control: hybrid_pruned | 27.5154 | 27.5154 | 1 |
| 4 | QFA control: classical | 29.3137 | 29.3137 | 1 |
| 5 | alignn | 33.4099 | 33.4099 | 1 |
| 6 | modnet_v0.1.12 | 34.7662 | 34.7662 | 1 |
| 7 | modnet_v0.1.10 | 40.2218 | 40.2218 | 1 |
| 8 | **QFA-Net (qt_hybrid_pruned)** | 47.7199 | 47.7199 | 1 |
| 9 | CrabNet | 60.8044 | 60.8044 | 1 |
| 10 | automatminer_expressv2020 | 67.5727 | 67.5727 | 1 |
| 11 | QFA control: rf_descriptors | 79.1761 | 79.1761 | 1 |
| 12 | cgcnnv2019 | 81.1553 | 81.1553 | 1 |
| 13 | rf | 82.3863 | 82.3863 | 1 |
| 14 | dummy | 337.1003 | 337.1003 | 1 |

QFA-Net rank: **4 of 9** (leaderboard models + QFA-Net main score; our own controls are listed but not counted).

## 2. Overview

| Task | Metric | Folds done | QFA-Net score (mean) | Best leaderboard model (same folds) | QFA-Net rank |
|---|---|---|---|---|---|
| jdft2d | MAE in meV/atom | 5 (0,1,2,3,4) | 35.3022 (qt_hybrid_pruned) | modnet_v0.1.12 33.1918 | 3 / 9 |
| steels | MAE in MPa | 3 (0,1,2) | 112.9253 (qt_hybrid_pruned) | modnet_v0.1.12 85.4713 | 7 / 8 |
| phonons | MAE in cm^-1 | 1 (0) | 47.7199 (qt_hybrid_pruned) | alignn 33.4099 | 4 / 9 |

## 3. Computational cost of QFA-Net

| Task | Tier | Seeds/fold | Trained numbers (direct) | Trained with Quantum-Train | Gates before pruning | Gates after pruning (per seed) | QT qubits | ORB extraction (one-time) | Training min per fold (mean; per fold) | QT materialised = QT model |
|---|---|---|---|---|---|---|---|---|---|---|
| jdft2d | tiny | 3 | 83515 | 1858 | 112 | 74 (mean 74.0) | 17 | 0.7 min (636 crystals) | 41.4 (38.4, 37.9, 40.4, 40.8, 49.3) | all true |
| steels | tiny | 3 | 4730-4794 | 948 | 112 | 74 (mean 74.0) | 13 | none (formula-only task) | 15.3 (11.9, 14.6, 19.5) | all true |
| phonons | small | 2 | 84091-84155 | 1864 | 112 | 80 (mean 80.0) | 17 | 1.5 min (1265 crystals) | 34.8 (34.8) | all true |

Hardware: 13th Gen Intel Core i5-13500H, 12 cores / 16 threads, 15.7 GB RAM, Windows 11, torch 2.15.0+cpu. CPU only, no GPU. Training minutes = all seeds and phases of one fold (classical, hybrid, pruning, Quantum-Train, controls), excluding the one-time ORB pass.

## 4. Tasks not run (or only partly run)

| Task | Reason |
|---|---|
| expt_gap, glass, expt_is_metal | Skipped for time: fold 0 needed ~45-50 min (first guess), only 45 min were left after jdft2d, steels and phonons |
| dielectric, log_gvrh, log_kvrh, perovskites | Skipped for time: fold 0 needed ~70-120 min, 45 min left |
| mp_gap, mp_is_metal, mp_e_form | Not in the plan (100k+ crystals; `--include-large`, better on HPC) |
| steels | 3 of 5 folds (folds 3, 4 skipped for time: ~19 min each, 11 min left) |
| phonons | 1 of 5 folds (folds 1-4 skipped for time: ~36 min each) |

No task FAILED. Run: `run_final.py --hours 5`, 02:39-07:29 on 10 Nov 2026; rerun the same command to continue
(finished folds are skipped).

## 5. Notes for reading the numbers

- jdft2d: classical, hybrid, hybrid_pruned and mlp_residual have 5-fold means within 0.11 meV/atom
  (27.16-27.27); per fold they differ by at most 0.80 (fold 1); on folds 3 and 4 they are identical, because phase B never beat the
  classical twin on validation and the twin (r = 0) was kept. So on jdft2d the quantum residual gives no measurable
  gain, and pruning 112 -> 74 gates costs nothing. Differences < ~2 meV/atom are noise.
- steels (formula only, no ORB): the circuit residual helped (hybrid_pruned 99.49 vs classical 109.15 MPa,
  better on 3/3 folds), while the same-size MLP residual never beat the twin (109.15 = classical).
- Quantum-Train (main score) trains 45x fewer numbers on jdft2d (1,858 vs 83,515) but is worse than direct
  training of the same pruned model (hybrid_pruned) on 8 of 9 folds (exception: steels fold 0, 124.33 vs 128.11);
  mean gap: jdft2d +8.04, steels +13.43, phonons +20.20.
- ORB-v2 is a frozen foundation model pretrained on public DFT structures; it was never trained on these targets
  (no test-label leakage), but its pretraining set may contain the same crystals.
- One fold (phonons) or three folds (steels) are not a full MatBench score; rank among the leaderboard is on
  the same folds only.
