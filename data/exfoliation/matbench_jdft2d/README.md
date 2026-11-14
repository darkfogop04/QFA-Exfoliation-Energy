# MatBench jdft2d (636 bulk crystals, exfoliation energy)

- Folder: `data/exfoliation/matbench_jdft2d/`
- Source: https://hackingmaterials.lbl.gov/matminer/dataset_summary.html (matminer.datasets)
- Paper: Dunn et al., npj Comput. Mater. 6, 138 (2020), https://doi.org/10.1038/s41524-020-00406-3
- Licence: CC-BY 4.0 (MatBench)
- Download method: matminer
- Has exfoliation labels in a common format: yes -> dataset.json.gz

## Notes
The benchmark version. Train/test with the official 5 folds (data/matbench_meta/folds.json); run_qfa_exfoliation.py does this automatically for this folder.

## Train and test QFA-Net on it

```
python experiments/run_qfa_exfoliation.py --dataset matbench_jdft2d --folds 0
```
