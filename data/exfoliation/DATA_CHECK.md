# Exfoliation datasets: what was verified and downloaded (13 Nov 2026)

`STATUS.md` (written by download_exfoliation_data.py) lists the downloads. This file records what each folder
actually contains after checking the files, and which ones can be used to train and test QFA-Net.

## Ready to train and test (`dataset.json.gz` present)

| Folder | Materials | Target, unit | Method | Structure | Made by |
|---|---|---|---|---|---|
| matbench_jdft2d | 636 | exfoliation energy, meV/atom | DFT OptB88vdW | bulk | download script (matminer) |
| jarvis_dft_3d | 813 (of 93,902) | exfoliation energy, meV/atom | DFT OptB88vdW | bulk | download script (jarvis-tools) |
| jarvis_dft_2d | 748 (of 1,103) | exfoliation energy, meV/atom | DFT OptB88vdW | monolayer | download script (jarvis-tools) |
| twodmatpedia | 4,527 (of 6,351) | exfoliation energy, meV/atom (-1,219 to 3,056) | DFT PBE (no vdW) | monolayer | download script (db.json.gz) |
| mc2d_mounet2018 | 1,694 (of 2,109 rows) | binding energy, meV/A^2 (1.5 to 222) | DFT DF2-C09 | monolayer (exfoliated) | convert_exfoliation_extra.py |
| mc2d_campi2022 | 2,848 (of 3,078) | binding energy, meV/A^2 (0.2 to 222) | DFT DF2-C09 | monolayer (as extracted) | convert_exfoliation_extra.py |
| massive_discovery_2025 | 2,988 | binding energy, meV/A^2 (2.6 to 1,180) | MatterSim ML potential (not DFT) | bulk | convert_exfoliation_extra.py |

**Used for training and testing (decision 13 Nov 2026: exfoliation energy only):** matbench_jdft2d,
jarvis_dft_3d, jarvis_dft_2d, twodmatpedia. The three binding-energy sets (mc2d_mounet2018, mc2d_campi2022,
massive_discovery_2025; interlayer binding energy per area, meV/A^2) are converted and kept, but not trained on.

The sets overlap (matbench_jdft2d is a subset of jarvis_dft_3d; mc2d_mounet2018 is contained in mc2d_campi2022;
many compounds appear in several sets). Each set is trained and tested only on its own random folds, never across
sets. Per-atom and per-area values are different quantities and are not mixed.

## Downloaded but no usable labels

| Folder | What is there | Why it cannot be used |
|---|---|---|
| vahdat_exfoliability | 281 MB zip: notebooks, trained RF model, 3D .xsf structures | The two label files (`export_structures_labeled.json`, `structures_df2_calc_with_bands_be.json`) in the archive are macOS alias stubs (182-190 bytes, `bplist00`), not data. Their labels come from MC2D vdW-DF2, which is covered by mc2d_mounet2018/campi2022. |
| alexandria_2d | 137,833 2D structures (jarvis-tools zip, 38 MB) | No exfoliation or binding energy field. |
| mc2d_campi2022 (9 GB AiiDA export) | not downloaded | Not needed: labels and structures come from `structure_2d.json` + `as_extracted_2d_structures.zip`. |

## Not available for download

| Folder | Finding |
|---|---|
| c2db | cmr.fysik.dtu.dk/c2db now points to https://2dhub.org/; no public file download found (full DB "on request"). |
| ashton_materialsweb | https://materialsweb.org fails (TLS error, unrecognized name); site appears down. |
| lm26_bjorkman | 26 RPA binding energies only in paper tables, no structures; too few to train or test on. Template CSV left in raw/. |
| no_public_data: X2DB | Experimental metadata (synthesis, characterization) of ~270-370 2D materials; no energies (2dhub.org/x2db). |
| no_public_data: ML2DDB | arXiv 2507.00584: no data link found. |
| no_public_data: Massive Discovery 2025 | Data found (Zenodo 10.5281/zenodo.17035156) -> moved to its own folder `massive_discovery_2025/`. |
