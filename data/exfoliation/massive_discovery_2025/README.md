# Massive Discovery of Low-Dimensional Materials (2025/2026): 2,988 2D materials with binding energies

- Folder: `data/exfoliation/massive_discovery_2025/`
- Source: Zenodo, https://doi.org/10.5281/zenodo.17035156 (record 18195019, file `databases.zip`, 71.7 MB)
- Paper: Bagheri, Berger, Komsa, Koskinen, "Massive Discovery of Low-Dimensional Materials from Universal
  Computational Strategy", arXiv:2509.20164; Chem. Mater. 38, 2395 (2026), doi 10.1021/acs.chemmater.5c03151
- Licence: CC-BY 4.0
- Download method: manual (Zenodo API), 13 Nov 2026
- Has exfoliation labels in a common format: yes -> dataset.json.gz (made by experiments/convert_exfoliation_extra.py)

## Notes
`raw/databases.zip` holds `2D_materials.json` (ASE-db JSON, 2,988 materials) and `screened_materials.json`
(153k screened bulk materials, no binding energy). Label: `E_exf`, the interlayer **binding energy in meV/A^2**
(per area, not per atom), computed with the **MatterSim universal ML potential, not DFT**. Structures are the
relaxed bulk crystals (Materials Project ids). The paper text mentions 3,057 2D materials; the deposited file has 2,988.

## Train and test QFA-Net on it

```
python experiments/run_qfa_exfoliation.py --dataset massive_discovery_2025 --folds 0
```
(The script prints "meV/atom"; for this dataset the unit is meV/A^2.)
