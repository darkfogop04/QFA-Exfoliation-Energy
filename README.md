# QFA-Net: Hybrid Quantum–Classical Model for Exfoliation Energy

**QFA-Net** (Quantum-compressed Foundation-Adapter Network) predicts the **exfoliation energy** of layered materials from their crystal structure. It combines three parts:
* a frozen, pretrained foundation model (ORB-v2) with a small learned adapter;
* an 8-qubit variational quantum circuit used as a residual head, with gate pruning;
* **Quantum-Train**, in which a 17-qubit circuit generates the network's weights, so only about 1.9k numbers are trained instead of about 84k.

M.Tech major project by **Deepak K Shegavi** and **Shashi Preetam**. Supervisor: **Dr. Arti Kashyap**, School of Physical Sciences (SPS).

### Results at a glance (test MAE in meV/atom, lower is better)

| Benchmark | QFA best variant | QFA-Net main (QT + pruned) | Best other model |
|---|---|---|---|
| JARVIS-Leaderboard, exfoliation energy (official split, 81 test) | **30.49 — #1 of 16** | 32.01 (#2) | matminer XGBoost 40.89 |
| MatBench jdft2d (official 5 folds) | **27.16** | 35.30 (#3 of 9) | MODNet 33.19 |
| JARVIS-DFT 3D (5 folds) | **27.52** | 38.46 | matminer XGBoost 38.30 (same folds) |
| JARVIS-DFT 2D (5 folds) | **29.26** | 37.04 | matminer XGBoost 47.48 (same folds) |
| 2DMatPedia (5 folds) | **40.29** | 57.10 | matminer RF 66.91 (same folds) |

**What the results show:**
* **The accuracy comes from the classical foundation-model features.** The quantum head matches its classical twin and an equal-size MLP control; it does not beat them.
* **The quantum parts cut training cost.** Pruning removes about 30 % of the circuit gates at no accuracy cost, and Quantum-Train trains 45× fewer parameters.

---

## Contents
1. [What we were trying to do](#1-what-we-were-trying-to-do)
2. [Project history: from HQ-GNN to QFA-Net](#2-project-history-from-hq-gnn-to-qfa-net)
3. [Architecture](#3-architecture)
4. [Classical part](#4-classical-part)
5. [Quantum part](#5-quantum-part)
6. [The model variants](#6-the-model-variants)
7. [Training procedure and fair-testing protocol](#7-training-procedure-and-fair-testing-protocol)
8. [Datasets](#8-datasets)
9. [How to use this model](#9-how-to-use-this-model)
10. [Project map: where to find what](#10-project-map-where-to-find-what)
11. [Results](#11-results)
12. [Conclusions, limitations and future work](#12-conclusions-limitations-and-future-work)
13. [References](#13-references)

---

## 1. What we were trying to do

**Exfoliation energy** is the energy per atom needed to peel one layer off a layered crystal:

  E_exf = E_monolayer / N_monolayer − E_bulk / N_bulk   (meV/atom)

A low value (below roughly 200 meV/atom) marks a material as a candidate for a 2D sheet, like graphene from graphite.

**Why predict it with machine learning?** Computing it with DFT is expensive:
* it needs two relaxed calculations, one for the bulk and one for the isolated layer with about 20 Å of vacuum;
* it needs a van der Waals functional (OptB88vdW);
* the cost grows with the cube of the system size.

A large 2D screening (Mounet et al. 2018) used about 3.7 million CPU core-hours, roughly a thousand per material. A trained model answers in milliseconds.

**What made it hard:**
* **The data are small.** MatBench jdft2d has only 636 materials. Deep graph networks overfit at this size, and on MatBench, descriptor models such as MODNet beat them.
* **Foundation models are accurate but expensive.** Large pretrained models (such as JMP) are the most accurate, but they fine-tune 30–235 million parameters on GPUs for every task. We had a laptop CPU.
* **The quantum question.** Can a small, classically simulated variational quantum circuit add anything to a materials model? Can quantum methods reduce the training cost?

**Our goal:** a model that is accurate on small exfoliation datasets and cheap enough to train on a laptop CPU, with an honest, controlled test of what the quantum parts contribute.

## 2. Project history: from HQ-GNN to QFA-Net

| Stage | Model | jdft2d test MAE (meV/atom, 5 folds) | Lesson |
|---|---|---|---|
| Mid-term | **HQ-GNN**: crystal graph → 3 × CGConv GNN → 8 numbers → 8-qubit VQC (PennyLane) → MLP | 57.2 ± 10.8 with a fair test | The GNN trained from scratch on 636 materials was the bottleneck. A classical-only twin scored 56.1, so the circuit added nothing. |
| Mid-term improvements | New circuit (RZ·RY·RZ masks, batched, 11× faster) + gate pruning + Quantum-Train | Pruned 53.8; Quantum-Train 58.6 with 2,286 trained parameters (32× fewer) | Pruning (104 → 61 gates) and Quantum-Train work, but accuracy stays limited by the weak features. |
| Final | **QFA-Net**: frozen foundation-model features + small adapter + quantum residual head + pruning + Quantum-Train | **27.2** (QT version 35.3) | Strong pretrained features fix the accuracy. The quantum parts are tested against matched classical controls. |

The mid-term HQ-GNN code is still in the repository; see the [project map](#10-project-map-where-to-find-what).

## 3. Architecture

```
 STRUCTURE PATH  (skipped for formula-only tasks)
 Crystal (N atoms)
   └─► ORB-v2 universal potential, FROZEN, run once per crystal
         → H: 16 depths × N atoms × 256                     (0 trained; ~25M frozen)
         └─► Depth mixer: 2 learned softmax mixes over the 16 depths → N × 512        (32 trained)
               └─► Linear 512 → 64                                                     (32,832)
                     └─► Graph adapter × 2: gated messages, 12 neighbours ≤ 8 Å → N × 64  (27,136)
                           └─► Pool: attention | mean | max → Linear 192 → 64  ⇒  g (64)   (12,417)

 FORMULA PATH
 Formula ─► Magpie 132 + symmetry 8 = 140 descriptors
          └─► mutual-information top-k (k = 4√n_train) ─► MLP k → 32 → 32  ⇒  c (32)   (~4k)

 JOIN   h = [g, c]  (96 numbers)
   ├─► Classical head 96 → 64 → 1                                ⇒  y_c          (6,273)
   └─► Quantum head: Linear 96 → 8, θ = π·tanh(·)
                     → 8-qubit circuit, 3 data re-uploading layers (RZ·RY·RZ + CNOT ring)
                     → 16 readouts ⟨Z_q⟩, ⟨Z_q Z_q+1⟩ → Linear 16 → 1 (starts at 0)  ⇒  r   (~860)

 OUTPUT   y = y_c + r

 COMPRESSION (training time only)
   • Gate pruning: 112 → 74–80 gates per circuit, validation-guarded
   • Quantum-Train: a 17-qubit circuit + small mapping MLP GENERATE the ≈83k classical weights;
     only ≈1.9k numbers are trained. Prediction afterwards is fully classical.
```

**Size of the model (exfoliation runs):**

| | Count |
|---|---|
| Trained directly | 83,739–84,411 (≈83.5k classical + ≈860 quantum head) |
| Trained with Quantum-Train | 1,858–1,896 |
| Circuit gates | 112 dense: 24 data RY + 64 trainable rotations + 24 CNOT |
| Circuit gates after pruning | 74 (tiny tier) or 80 (small tier) |

## 4. Classical part

1. **Frozen foundation features (ORB-v2).** ORB-v2 (Orbital Materials, Apache-2.0) is a universal interatomic potential pretrained on large DFT datasets. It was never trained on exfoliation energies.
   * Forward hooks copy every atom's 256-number vector after the encoder and after each of its 15 message-passing layers, giving 16 depths.
   * No gradient ever reaches ORB.
   * The vectors are computed once per dataset and cached (`data/cache/qfa_<task>_orb_v2_atoms_v1.npz`, float16).
2. **Depth mixer (our addition).** Which ORB depth carries the most useful information changes from task to task: HackNIP (2025) found layer 11 best for exfoliation energy and layer 3 best for phonons. Instead of searching per task, we learn it:
   * m_k = Σ_l softmax(α_k)_l · LayerNorm(H[l]), for k = 1, 2.
   * Mix 1 starts on the deep half of the layers, mix 2 on the shallow half.
   * Only 32 numbers are trained. The learned weights are saved in the results (`depth_weights`).
3. **Graph adapter (our addition).**
   * Our own neighbour graph: the 12 nearest atoms within 8 Å (ORB itself looks only 6 Å).
   * Every periodic image is its own edge, and each distance is expanded into 16 Gaussian bins.
   * Two gated message-passing layers: h_i ← LayerNorm(h_i + Σ_j σ(W_g[h_i, h_j, e_ij]) ⊙ (W_v h_j) / √deg_i).
4. **Pooling:** an attention-weighted sum, the mean and the max over atoms, then Linear 192 → 64, giving **g**.
5. **Descriptor branch (MODNet idea):**
   * Magpie features (22 element properties × 6 statistics; our NumPy re-implementation of matminer's preset, checked by hand on MoS₂) plus crystal system (7) and space group.
   * The top k = 4√n_train features are picked by mutual information, using the training fold only and dropping near-duplicates with |corr| > 0.95.
   * An MLP maps them to **c**.
   * Formula-only tasks use this branch alone.
6. **Classical head:** h = [g, c] → 96 → 64 → 1, giving **y_c**.

## 5. Quantum part

### 5.1 Circuit (the "hybrid" head)
* **Encoding:** u = Linear(h) gives 8 numbers; θ = π·tanh(u) puts each angle in (−π, π). One angle per qubit, applied as RY(θ_q)|0⟩.
  * Worked example: θ = 0.8 → 0.921|0⟩ + 0.389|1⟩ → ⟨Z⟩ = 0.849 − 0.151 = 0.697.
* **Layers:** 8 qubits, 3 layers. Each layer applies:
  * RY(θ) on every qubit (data re-uploading);
  * trainable RZ·RY·RZ on every qubit;
  * a CNOT ring.
* **Simulation:** 2⁸ = 256 amplitudes, simulated exactly (state vector) in our own batched PyTorch code (`src/qfa/quantum_torch.py`). It is tested against a NumPy reference with adjoint gradients (`src/qfa/lab/qsim.py`).
* **Readout:** ⟨Z_q⟩ (8 values) and ⟨Z_q Z_q+1⟩ (8 values), 16 numbers in total, followed by a Linear layer that gives **r**.
  * The final RZ gates of the last layer cannot change any Z-type readout, so they are removed from the start (72 → 64 trainable angles).
* **Residual that starts at zero:** y = y_c + r, with the output layer of r initialised to 0. At epoch 0 the hybrid is therefore *exactly* the classical model (its "twin"). It keeps a change only if validation error improves, so it can never be worse than classical on validation.

### 5.2 Gate pruning
1. Score every active rotation by **ablation**: set its angle to 0, which turns it into an identity gate, and measure the validation error.
2. Remove the 25 % of rotations whose removal hurts least, then fine-tune.
3. Repeat.
4. Undo the last round if validation error rises more than 2 % above the dense hybrid.
5. Stop at the tier limit (60 / 50 / 40 / 30 % of the rotations).

Measured effect: 112 → 74 gates (tiny tier) or 112 → 80 gates (small tier), with the error change inside run-to-run noise.

### 5.3 Quantum-Train (QT)
Based on Liu et al. 2024. The P classical weights are not trained directly; a quantum circuit generates them.
* **Generator circuit:** n = ⌈log₂ P⌉ qubits (17 for the full model), with 8 layers of RY + a CNOT chain. Its output is 2ⁿ probabilities p_k.
* **Mapping MLP:** a shared MLP turns (the bits of k, 2ⁿ·p_k − 1) into weight k, multiplied by one learned scale per tensor.
* **What is trained:** 136 circuit angles, 593 mapping numbers, the scales, and the small parts kept direct (normalisation layers, the quantum head). That is about **1.9k numbers instead of about 84k**.
* **Materialise:** after training, the generated weights are copied into the normal network. Prediction needs no quantum computer.
  * A check in every run confirms that the copied model gives identical predictions (`qt_materialised_equal = true` in every result file).
* QT reduces **trained** parameters, not the size of the model used at prediction time.

## 6. The model variants

All variants come from the **same run** and the same seeds, and are tested on the same test materials. This is what makes the quantum comparison fair.

| Variant (key in result files) | What it is | Trained numbers | Circuit gates | Role |
|---|---|---|---|---|
| **Classical** (`classical`) | ORB features + depth mixer + adapter + descriptor branch + classical head, with r fixed at 0 | ≈83.5k | none | Classical twin, the baseline for every quantum claim |
| **Hybrid** (`hybrid`) | Classical + 8-qubit circuit residual head | ≈84.3k | 112 | Does a quantum head help? |
| **Hybrid pruned** (`hybrid_pruned`) | Hybrid after ablation gate pruning | ≈84.3k | 74–80 | Same accuracy with fewer gates? |
| **MLP control** (`mlp_residual`) | Same residual design, but a classical MLP of the same size replaces the circuit | ≈84.3k | none | Is any gain quantum, or just "more head"? |
| **QFA-Net main** (`qt_hybrid_pruned`) | **Pruned hybrid trained with Quantum-Train** | **≈1.9k** | 74–80 | The full quantum-compressed model |
| **RF on our descriptors** (`rf_descriptors`) | Random forest on the Magpie + symmetry features | – | – | Sanity check against MatBench's RF |
| Median / Dummy | Predict the training median / mean | – | – | Floor every model must beat |

## 7. Training procedure and fair-testing protocol

For each fold and each seed:

| Phase | What is trained | Rule |
|---|---|---|
| A classical | mixer, adapter, pooling, descriptor MLP, classical head (r = 0) | AdamW lr 1e-3, L1 loss (BCE for classification), early stopping on validation |
| B quantum residual | circuit head (lr 1e-2) + classical part (lr 2e-4) | epoch 0 = classical twin; best validation epoch kept |
| C pruning | circuit rotations | ablation rounds, validation-guarded |
| D Quantum-Train | 17-qubit generator + mapping MLP + kept parts; pruning mask reused | tiny/small tiers only |

Predictions of all seeds are averaged. Every training stage has a time cap: 8 min for the tiny tier and 12 min for the small tier.

**Size tiers** (set from the training-set size, in `src/qfa/lab/experiment.py`):

| Tier | Training size | Seeds per fold | Prune limit | QT |
|---|---|---|---|---|
| tiny | ≤ 1,000 | 3 | 60 % | on |
| small | ≤ 10,000 | 2 | 50 % | on |
| medium | ≤ 30,000 | 1 | 40 % | off |
| large | > 30,000 | 1 | 30 % | off |

**Fairness rules, followed in every result below:**
* **The test fold is never used for any decision** (early stopping, pruning, feature selection, tuning). Validation is 10 % of the training part.
* **Each test fold is used once**, at the end.
* **MatBench tasks:** the official MatBench folds. Leaderboard models are compared on the same folds, using their published per-fold scores (`data/matbench_meta/`).
* **JARVIS-Leaderboard:** the official split file (650 train + 81 val, 81 test), so our score sits next to the 14 published models on identical test materials.
* **Other exfoliation datasets:** 5 random folds (seed 42), saved in `data/exfoliation/<dataset>/folds.json`. The same folds are used for every model, including the re-run open-source baselines.
* **Noise:** differences below about 2 meV/atom are treated as run-to-run noise.

## 8. Datasets

| Dataset | Materials | Structures | Target | Source |
|---|---|---|---|---|
| **MatBench jdft2d** | 636 | bulk | exfoliation energy, meV/atom (DFT OptB88vdW) | MatBench v0.1 |
| **JARVIS-DFT 3D** | 813 | bulk | exfoliation energy, meV/atom (OptB88vdW); jdft2d is a subset | jarvis-tools |
| **JARVIS-DFT 2D** | 748 | monolayer | exfoliation energy, meV/atom (OptB88vdW) | jarvis-tools |
| **2DMatPedia** | 4,527 | monolayer | exfoliation energy, meV/atom (PBE, no vdW) | 2dmatpedia.org |
| MatBench steels, phonons, expt_gap, glass | 312–5,680 | formula / bulk | yield strength, phonon peak, band gap, glass formation | MatBench v0.1 |

Each set is trained and tested only on its own folds, never mixed. Interlayer binding-energy sets (MC2D and massive_discovery_2025, in per-area units) are included as converted files, but were not used for training, because they measure a different quantity. Details per dataset: `data/exfoliation/DATA_CHECK.md` and each dataset folder's `README.md`.

## 9. How to use this model

### 9.1 Requirements
* Python 3.13, CPU only. All results were produced on an Intel i5-13500H laptop (16 threads, 15.7 GB RAM) running Windows 11. A `--device cuda` option exists but was not used or tested for these results.
* About 2 GB of free disk space for the feature caches (more if you download every exfoliation dataset).

### 9.2 Install
```bash
git clone <this repository>
cd QFA-Exfoliation-Energy
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps matbench==0.6     # its scikit-learn pin conflicts with the rest
```
* The first ORB-v2 run downloads its weights (about 100 MB).
* MatBench downloads each task once (internet is needed the first time).

### 9.3 Check that everything works
```bash
pytest tests/test_qfa.py tests/test_qfa_lab.py -q        # 14 passed
python experiments/check_qfa_setup.py                     # every line [OK]
python experiments/run_qfa.py --task jdft2d --folds 0 --quick    # ~10 min end-to-end smoke test
```
`--quick` results are written as `fold0_quick.json` and ignored by every summary.

### 9.4 Train and test on a MatBench task (official folds)
```bash
python experiments/run_qfa.py --task jdft2d                  # all 5 folds (~40 min per fold on a laptop CPU)
python experiments/run_qfa.py --task jdft2d --folds 0,1      # selected folds
python experiments/run_qfa.py --task phonons --seeds 2 --max-minutes 10
python experiments/run_qfa.py --task jdft2d --summary        # our scores vs the leaderboard on the same folds
python experiments/run_qfa.py --task all --export results/qfa/final_results.json
python experiments/run_final.py --hours 5.5                  # time-boxed run over many tasks (resumable)
```

`run_qfa.py` options:
* `--task` is a MatBench name without `matbench_` (jdft2d, steels, phonons, expt_gap, glass, dielectric, log_gvrh, …) or `all`.
* `--ablation` is one of `last_layer`, `no_adapter`, `no_desc`, for classical ablation studies.
* `--backbone none` runs without ORB: formula and symmetry features plus the quantum head.
* `--device` takes `cpu` or `cuda`.

### 9.5 Train and test on the exfoliation datasets
```bash
python experiments/download_exfoliation_data.py --only jarvis_dft_3d,jarvis_dft_2d,twodmatpedia
python experiments/run_qfa_exfoliation.py --list                         # datasets ready to use
python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_2d --folds 0 --quick
python experiments/run_qfa_exfoliation.py --dataset twodmatpedia         # all 5 folds
python experiments/run_exfoliation_all.py --hours 7 --folds 0,1,2,3,4    # every dataset, time-boxed, resumable
python experiments/exfoliation_results_table.py                          # -> results/qfa_exfoliation/RESULTS_TABLE.md
```
The `dataset.json.gz` and `folds.json` files of the datasets used here are already in the repository, so the training commands work without downloading. The raw downloads are only needed to rebuild them.

### 9.6 Use the model on your own dataset
1. Create a folder `data/exfoliation/<my_dataset>/`.
2. In it, create `dataset.json.gz`: a gzipped JSON list with one entry per material.
   ```json
   [{"id": "mat-001", "formula": "MoS2", "structure": { ...pymatgen Structure.as_dict()... }, "target": 76.3}, ...]
   ```
   * `target` can be any regression property; units are your choice.
   * Duplicate `id`s are dropped.
3. Run:
   ```bash
   python experiments/run_qfa_exfoliation.py --dataset my_dataset            # 5 random folds (seed 42)
   python experiments/run_qfa_exfoliation.py --dataset my_dataset --summary  # mean test MAE of every variant
   ```
   * The folds are created once and saved to `data/exfoliation/my_dataset/folds.json`, so every later run and model uses the same split.
   * The size tier (seeds, pruning limit, Quantum-Train on/off) is chosen automatically from the dataset size.
   * For very large sets, `--max-n 20000` uses a random subset.

The pipeline trains and tests the model; it does not save weights. Each run trains all variants from scratch and reports their test scores.

### 9.7 Reproduce the comparison with the best open-source models
```bash
python experiments/run_jarvis_leaderboard_split.py       # QFA on the official JARVIS-Leaderboard split (~50 min)
python experiments/run_matminer_rf.py --model rf         # matminer RF (JARVIS-Leaderboard #2 recipe)
python experiments/run_matminer_rf.py --model xgb        # matminer XGBoost (JARVIS-Leaderboard #1 recipe)
python experiments/exfoliation_results_table.py          # adds both baselines to the per-dataset tables
```

### 9.8 Reading the output
Each fold writes one JSON file, `results/<run>/<dataset>/fold<k>.json`:

| Key | Meaning |
|---|---|
| `test` | test score of every variant: `classical`, `hybrid`, `hybrid_pruned`, `mlp_residual`, `qt_hybrid_pruned`, `rf_descriptors` |
| `per_seed[i].params` / `params_pruned` | trained numbers (classical part, quantum head, total) |
| `per_seed[i].gates_dense` / `gates_pruned` | circuit gates before and after pruning |
| `per_seed[i].prune_history` | validation error after every pruning round |
| `per_seed[i].qt_trained`, `qt_qubits`, `qt_materialised_equal` | Quantum-Train size, qubits, materialisation check (must be true) |
| `per_seed[i].depth_weights` | learned depth-mixer weights over the 16 ORB layers |
| `per_seed[i].sec_*`, `seconds` | time per phase and per fold |

Scores are MAE for regression (lower is better) and ROC-AUC for classification (higher is better).

### 9.9 Practical tips
| Situation | What to do |
|---|---|
| A long run was interrupted | Run the same command again: finished folds are skipped |
| Out of memory | Run one dataset at a time with `--seeds 1`, and close other programs |
| A fold is too slow | Lower `--max-minutes` (the cap per training stage) |
| `orb-models` install fails on `dm-tree` | `pip install dm-tree --only-binary :all:`, then `pip install orb-models --no-deps` |
| Windows multiprocessing errors | Run scripts from a terminal; every entry script has a `__main__` guard |
| `qt_materialised_equal` is false | The Quantum-Train numbers of that run are not trustworthy; rerun it |

## 10. Project map: where to find what

```
QFA-Exfoliation-Energy/
│
├── README.md                          this document
├── requirements.txt                   QFA-Net environment (exact versions used for all results)
├── requirements-legacy.txt            mid-term HQ-GNN environment (PennyLane, torch-geometric)
├── pytest.ini                         test settings
├── .gitignore                         excludes feature caches and raw downloads
│
├── src/                               ─────────── SOURCE CODE ───────────
│   ├── qfa/                           ★ QFA-Net (final model, PyTorch)
│   │   ├── backbone.py                ORB-v2 loading, per-depth forward hooks, embedding cache
│   │   ├── data.py                    descriptors, image-aware 12-neighbour graph, task caches, batching
│   │   ├── model.py                   QFANet: DepthMixer, GatedAdapterLayer, pooling, classical/quantum heads
│   │   ├── quantum_torch.py           batched state-vector circuit (8 qubits) + Quantum-Train generator
│   │   ├── train.py                   phases A–D (classical, hybrid, pruning, QT), controls, one fold
│   │   └── lab/                       NumPy-only reference implementation (no PyTorch)
│   │       ├── featurize.py           Magpie composition + symmetry features
│   │       ├── magpie_data/           element property tables used by featurize.py
│   │       ├── qsim.py                circuit simulator with exact adjoint gradients
│   │       ├── qt.py                  Quantum-Train in NumPy
│   │       ├── nn.py                  small MLP with hand-written backprop + AdamW
│   │       ├── data.py                MatBench targets/compositions for the lab runs
│   │       └── experiment.py          one lab fold; size tiers (TIERS), metrics, feature selection
│   │
│   ├── models/                        mid-term HQ-GNN: gnn.py (CGConv), quantum_layer.py (PennyLane VQC),
│   │                                  hybrid_model.py, classical_baseline.py
│   ├── optimization/                  mid-term: pruning.py, quantum_train.py, size_policy.py
│   ├── training/                      mid-term: trainer.py, matbench_runner.py, metrics.py
│   ├── data/                          mid-term: crystal_graph.py, graph_cache.py, dataset.py, featurizers.py
│   └── utils/                         config.py, seed.py
│
├── experiments/                       ─────────── RUNNABLE SCRIPTS ───────────
│   ├── check_qfa_setup.py             5-minute pre-flight check of the installation
│   ├── run_qfa.py                     QFA-Net on MatBench tasks (official folds); --summary, --export
│   ├── run_final.py                   time-boxed, resumable run over many MatBench tasks (writes STATUS.md)
│   ├── run_qfa_exfoliation.py         QFA-Net on one exfoliation dataset (or your own dataset)
│   ├── run_exfoliation_all.py         time-boxed, resumable run over all exfoliation datasets
│   ├── run_jarvis_leaderboard_split.py QFA-Net on the official JARVIS-Leaderboard split
│   ├── run_matminer_rf.py             open-source baselines: matminer RF / XGBoost (--model rf|xgb)
│   ├── exfoliation_results_table.py   per-dataset result tables (all variants + baselines)
│   ├── download_exfoliation_data.py   downloads JARVIS 3D/2D, 2DMatPedia and other exfoliation sets
│   ├── convert_exfoliation_extra.py   converts MC2D / massive_discovery_2025 files to dataset.json.gz
│   ├── run_qfa_lab.py                 NumPy lab runs (formula + symmetry only)
│   ├── summarize_qfa_lab.py           summary of the lab runs
│   └── run_training.py, run_baseline.py, run_optimization.py, build_graph_cache.py   (mid-term HQ-GNN)
│
├── configs/                           mid-term HQ-GNN configs: default.yaml, optimize.yaml, quick.yaml
│
├── tests/                             test_qfa.py, test_qfa_lab.py (QFA-Net); the others test HQ-GNN
│
├── data/                              ─────────── DATA ───────────
│   ├── matbench_meta/                 official MatBench fold ids (folds.json), leaderboard per-fold
│   │                                  scores, MatBench task metadata (*.json.bz2)
│   ├── exfoliation/
│   │   ├── README.md, DATA_CHECK.md   what every dataset contains and which ones are usable
│   │   ├── jarvis_dft_3d/             dataset.json.gz, folds.json, jarvis_leaderboard_split.json
│   │   ├── jarvis_dft_2d/             dataset.json.gz, folds.json
│   │   ├── twodmatpedia/              dataset.json.gz, folds.json
│   │   ├── matbench_jdft2d/           dataset.json.gz
│   │   ├── mc2d_*/, massive_discovery_2025/   binding-energy sets (converted, not used for training)
│   │   └── c2db/, ashton_materialsweb/, lm26_bjorkman/, no_public_data/   notes on unavailable data
│   └── cache/                         feature caches (not in the repository; built on the first run)
│
└── results/                           ─────────── ALL MEASURED RESULTS ───────────
    ├── qfa/                           MatBench runs
    │   ├── <task>/fold<k>.json        jdft2d (5 folds), steels (3), phonons, expt_gap, glass (1 each)
    │   ├── RESULTS_TABLE.md           every task vs the leaderboard on the same folds + cost table
    │   ├── final_results.json, SUMMARY.txt, STATUS*.md, run_final_log*.txt
    │   └── CHANGES.md                 log of every code change made during the final runs
    ├── qfa_exfoliation/               exfoliation datasets
    │   ├── <dataset>/fold<k>.json     JARVIS 3D, JARVIS 2D, 2DMatPedia (5 folds each)
    │   ├── <dataset>/matminer_rf.json, matminer_xgboost.json   re-run open-source baselines
    │   ├── jarvis_dft_3d_leaderboard/test.json                 QFA on the official JARVIS split
    │   ├── RESULTS_TABLE.md           per-fold tables of every model
    │   ├── COMPARISON_BEST_MODELS.md  QFA vs the best open-source models, per dataset
    │   └── STATUS*.md, run_log_*.txt  run status and logs
    └── qfa_lab/                       NumPy lab runs (jdft2d, steels; formula + symmetry only)
```

## 11. Results

All scores are on test data never used for training or any decision. MAE is in meV/atom unless stated; lower is better. "QFA best" means the best of the four direct variants, which are within noise of each other.

### 11.1 JARVIS-Leaderboard: exfoliation energy (official split, same 81 test materials)

| Rank | Model | MAE |
|---|---|---|
| **1** | **QFA classical / hybrid / hybrid_pruned / mlp_residual** | **30.49** |
| **2** | **QFA-Net main (QT + pruned, 1,858 trained)** | **32.01** |
| 3 | matminer XGBoost (previous #1) | 40.89 |
| 4 | matminer RF | 42.76 |
| 5 | kgcnn CGCNN | 45.76 |
| 6 | kgcnn DimeNet++ | 46.15 |
| 7 | kgcnn coNGN | 46.27 |
| 8 | kgcnn coGN | 47.70 |
| 9 | kgcnn SchNet | 48.30 |
| 10 | matminer LightGBM | 49.51 |
| 11 | Matformer | 51.66 |
| 12 | ALIGNN | 52.70 |
| 13 | CGCNN | 52.70 |
| 14 | CFID | 62.12 |
| 15 | CFID-chem | 63.77 |
| 16 | kgcnn MEGNet | 68.24 |

Scores of other models are published on [pages.nist.gov/jarvis_leaderboard](https://pages.nist.gov/jarvis_leaderboard/AI/SinglePropertyPrediction/dft_3d_exfoliation_energy). On this split the four direct QFA variants gave identical predictions, because the quantum residual stayed at zero.

### 11.2 MatBench jdft2d (official 5 folds)

| Rank | Model | MAE |
|---|---|---|
| – | QFA mlp_residual / hybrid / classical / hybrid_pruned | **27.16 / 27.18 / 27.23 / 27.27** |
| 1 | MODNet v0.1.12 (MatBench #1) | 33.19 |
| 2 | MODNet v0.1.10 | 34.54 |
| **3** | **QFA-Net main (QT + pruned)** | **35.30** |
| 4 | AutoMatminer Express | 39.85 |
| 5 | ALIGNN | 43.42 |
| 6 | CrabNet | 45.61 |
| 7 | CGCNN | 49.24 |
| 8 | Random Forest | 50.04 |
| 9 | Dummy | 67.29 |

### 11.3 Exfoliation datasets: 5 folds, all variants and open-source baselines on the same folds

| Model | JARVIS 3D (813) | JARVIS 2D (748) | 2DMatPedia (4,527) |
|---|---|---|---|
| QFA classical | **27.52 ± 3.43** | 29.50 ± 6.00 | 40.62 ± 3.64 |
| QFA hybrid | 27.64 ± 3.29 | 29.43 ± 5.86 | 40.46 ± 3.74 |
| QFA hybrid_pruned | 27.65 ± 3.29 | **29.26 ± 6.02** | **40.29 ± 3.50** |
| QFA mlp_residual (control) | 27.54 ± 3.43 | 29.37 ± 5.97 | 40.37 ± 3.47 |
| **QFA-Net main (QT + pruned)** | 38.46 ± 5.97 | 37.04 ± 6.30 | 57.10 ± 7.16 |
| matminer XGBoost (JARVIS #1 recipe, re-run) | 38.30 ± 3.88 | 47.48 ± 9.85 | 67.10 ± 5.95 |
| matminer RF (JARVIS #2 recipe, re-run) | 39.06 ± 3.80 | 49.96 ± 10.10 | 66.91 ± 6.65 |
| RF on our descriptors | 45.12 ± 3.14 | 52.15 ± 8.90 | 72.80 ± 5.92 |
| Median predictor | 53.26 ± 6.67 | 58.38 ± 6.57 | 102.08 ± 4.48 |
| Dummy (training mean) | 62.79 ± 3.78 | 71.07 ± 4.40 | 119.56 ± 3.01 |

Each cell is the mean ± std over the 5 folds.
* **QFA best vs XGBoost:** QFA's best variant beats XGBoost on **5/5 folds** of every dataset, by 28–40 %.
* **QFA-Net main (QT) vs XGBoost:** QT beats XGBoost on 5/5 folds of JARVIS 2D and 2DMatPedia, and ties on JARVIS 3D (3/5 folds).
* **Baseline re-runs are faithful:** on the official split our re-runs gave 42.71 (XGBoost) and 44.33 (RF), against the published 40.89 and 42.76.
* **Caveat for the matminer baselines:** on monolayers, matminer's structure features fail for most materials because of the vacuum layer. This is how the published recipe behaves; the features are imputed.

Per-fold tables: `results/qfa_exfoliation/RESULTS_TABLE.md`. Leaderboard comparison: `results/qfa_exfoliation/COMPARISON_BEST_MODELS.md`.

### 11.4 Other MatBench tasks

These are partial runs: 3 folds for steels, fold 0 only for the others.

| Task (metric) | QFA best variant | QFA-Net main (QT) | Best leaderboard model (same folds) | QT rank |
|---|---|---|---|---|
| phonons (MAE, cm⁻¹), fold 0 | **27.18** (mlp) / 27.48 (hybrid) | 47.72 | ALIGNN 33.41 | 4 / 9 |
| glass (ROC-AUC ↑), fold 0 | 0.933 (hybrid_pruned) | 0.913 | MODNet 0.974 | 2 / 6 |
| steels (MAE, MPa), folds 0–2 | 99.49 (hybrid_pruned) | 112.93 | MODNet 85.47 | 7 / 8 |
| expt_gap (MAE, eV), fold 0 | 0.393 | 0.488 | MODNet 0.312 | 10 / 11 |

Steels, expt_gap and glass are formula-only tasks: no structure, so no ORB features. QFA is weaker there, which supports the conclusion that its strength comes from the foundation-model structure features. Full tables: `results/qfa/RESULTS_TABLE.md`.

### 11.5 Effect of each quantum component (paired, same runs)

| Comparison | Result |
|---|---|
| Hybrid vs classical twin | Within 0.8 meV/atom on every fold of jdft2d and the three exfoliation sets (noise). The circuit gives no measurable gain or loss. |
| Hybrid vs MLP control of the same size | Equal within noise, so no quantum-specific effect on accuracy. |
| Pruned vs dense hybrid | 112 → 74–80 gates (−29 to −34 %) at equal accuracy. |
| QT vs direct training | 45× fewer trained numbers (≈84k → ≈1.9k). The cost is 8–17 meV/atom MAE on the 5-fold sets, and only +1.5 on the JARVIS-Leaderboard split. |
| QT with a circuit vs without one (NumPy lab, jdft2d, 3 seeds) | 45.93 vs 49.12: the circuit generator is better on 4/5 folds. This is the clearest quantum-specific signal. |

### 11.6 Computational cost (laptop CPU, no GPU)

| Dataset | Training per fold (all variants, all seeds) | Seeds |
|---|---|---|
| MatBench jdft2d | ≈41 min (+0.7 min one-time ORB pass) | 3 |
| JARVIS 3D / 2D | ≈43 / 39 min | 3 |
| 2DMatPedia | ≈49 min (+≈9 min one-time ORB pass) | 2 |
| JARVIS-Leaderboard split | 54 min | 3 |

By comparison, a DFT exfoliation calculation takes about 1,000 CPU core-hours per material. A trained QFA model predicts in milliseconds.

## 12. Conclusions, limitations and future work

**Conclusions**
1. **QFA-Net is accurate on exfoliation energy.** It ranks #1 on the JARVIS-Leaderboard (30.49 vs 40.89), beats MatBench's best jdft2d model (27.2 vs 33.2), and beats the best open-source recipes on every fold of three more datasets.
2. **The accuracy is classical.** It comes from frozen ORB-v2 foundation features, adapted with only about 84k trained numbers on a CPU.
3. **The quantum circuit matches its classical twin and an equal-size MLP, but does not beat them.** We report no quantum advantage in accuracy.
4. **Quantum methods do reduce cost.** Pruning removes about a third of the gates for free. Quantum-Train trains 45× fewer numbers and still beats every published JARVIS-Leaderboard model (32.01). On the 5-fold sets it beats the open-source recipes on 2 of 3 datasets and ties on the third.

**Limitations**
* The circuits are classically simulated (8 qubits for the head, 17 for QT). No real quantum hardware was used.
* The JARVIS 2D and 2DMatPedia comparisons use re-run open-source recipes. Those datasets have no public leaderboard, and GNN baselines (ALIGNN, CGCNN) were not re-run on them.
* The remaining MatBench tasks have 1–3 folds only. Medium and large tasks (dielectric, elastic moduli, perovskites, mp_*) were not run.
* QT reduces trained parameters, not the size of the model used at prediction time.
* The pipeline does not save trained weights; every run trains from scratch.

**Future work**
* Run all 5 folds of every MatBench task (ideally on an HPC).
* Run noise-aware or hardware runs of the 8-qubit head.
* Try QT with fewer qubits through batched parameter generation.
* Re-run GNN baselines on the 2D datasets.
* Save and load trained models for direct prediction.
* Fine-tune the last ORB layers on the large tasks.

## 13. References

MatBench (Dunn et al. 2020) · JARVIS-Leaderboard (Choudhary et al. 2024) · ORB (Neumann et al. 2024) · HackNIP (Kim, Park & Li 2025) · MODNet (De Breuck et al. 2021) · ALIGNN (Choudhary & DeCost 2021) · CGCNN (Xie & Grossman 2018) · coGN/coNGN (Ruff et al. 2024) · Magpie (Ward et al. 2016) · matminer (Ward et al. 2018) · Quantum-Train (Liu et al. 2024) · data re-uploading (Pérez-Salinas et al. 2020) · JMP (Shoghi et al. 2024) · 2DMatPedia (Zhou et al. 2019) · Mounet et al. 2018 (2D exfoliation screening).
