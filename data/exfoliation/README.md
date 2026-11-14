# Exfoliation-energy datasets

One folder per dataset. Each has a `README.md` (source, paper, licence, notes) and a `raw/` folder for the files
exactly as downloaded. Labelled datasets also get `dataset.json.gz` in one common format, which
`experiments/run_qfa_exfoliation.py` trains and tests QFA-Net on.

## 1. Download (on your laptop, needs internet)

```
D:\MTP\venv_qmat\Scripts\Activate.ps1
pip install -U jarvis-tools matminer requests
python experiments/download_exfoliation_data.py                  # all except the ~9 GB MC2D 2022 export
python experiments/download_exfoliation_data.py --include-huge   # also the 9 GB file
python experiments/download_exfoliation_data.py --only jarvis_dft_3d,twodmatpedia   # just some
```
At the end it writes `STATUS.md` here: what was downloaded, what failed, what needs a manual step.
Re-running skips files already downloaded.

## 2. Train and test

```
python experiments/run_qfa_exfoliation.py --list                                   # labelled datasets ready
python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d --folds 0 --quick  # smoke test
python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d                    # 5 folds
python experiments/run_qfa_exfoliation.py --dataset twodmatpedia --max-n 3000        # subset of a big set
python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d --summary          # MAE table
```
Score: MAE in meV/atom (lower is better), all six QFA variants (classical, hybrid, hybrid_pruned,
qt_hybrid_pruned, mlp_residual, rf_descriptors). `matbench_jdft2d` uses the official MatBench folds; the others use
5 random folds (seed 42) saved as `folds.json` in their folder.

## 3. The folders

| Folder | What | Labels for training | How it downloads |
|---|---|---|---|
| `jarvis_dft_3d` | JARVIS bulk crystals with exfoliation energy | yes (meV/atom, OptB88vdW) | jarvis-tools |
| `jarvis_dft_2d` | JARVIS monolayers | if the field exists | jarvis-tools |
| `matbench_jdft2d` | MatBench benchmark, 636 crystals | yes | matminer |
| `twodmatpedia` | 2DMatPedia, 6,351 monolayers | yes (PBE, eV/atom x 1000) | web page JSON |
| `mc2d_mounet2018` | Materials Cloud 2D, Mounet 2018 (~953 MiB) | raw only (binding energy per area) | Materials Cloud API |
| `mc2d_campi2022` | Materials Cloud 2D, Campi 2022 (~9 GB AiiDA) | raw only | Materials Cloud API, `--include-huge` |
| `vahdat_exfoliability` | Vahdat et al. ML exfoliability (~281 MiB zip) | raw only | Materials Cloud API |
| `alexandria_2d` | Alexandria 2D, ~200k structures | none (no exfoliation labels) | jarvis-tools |
| `c2db` | C2DB (DTU) | raw only | manual (see its README) |
| `ashton_materialsweb` | Ashton 2017, 826 monolayers | raw only | manual |
| `lm26_bjorkman` | LM26, 26 RPA binding energies | fill the CSV template from the paper | manual |
| `no_public_data` | Massive Discovery 2025, X2DB, ML2DDB | none found | manual |

## 4. Take care

- **The sets overlap.** `matbench_jdft2d` is a subset of `jarvis_dft_3d`, and many 2DMatPedia, C2DB and MC2D
  materials are the same compounds. Never train on one set and test on another without removing shared materials
  (match by formula and structure), or the test score is too optimistic.
- **Different methods give different numbers.** JARVIS uses OptB88vdW, 2DMatPedia PBE, MC2D DF2-C09/rVV10 per area.
  Do not mix them in one training set without saying so.
- Licences differ; check each README before sharing the data.
