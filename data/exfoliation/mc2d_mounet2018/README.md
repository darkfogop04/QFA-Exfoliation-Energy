# Materials Cloud 2D, Mounet et al. 2018 (~1,825 easily exfoliable compounds)

- Folder: `data/exfoliation/mc2d_mounet2018/`
- Source: https://archive.materialscloud.org/records/rbqe0-q3717 (~953 MiB)
- Paper: Mounet et al., Nat. Nanotechnol. 13, 246 (2018), https://doi.org/10.1038/s41565-017-0035-5
- Licence: CC-BY 4.0 (see the record)
- Download method: mcloud
- Has exfoliation labels in a common format: no (raw files only)

## Notes
Binding energies (meV/A^2, DF2-C09 and rVV10) rather than meV/atom; the record has simple tables plus AiiDA exports. Raw files only; inspect the tables before training (see README in raw/).

## Train and test QFA-Net on it

No common-format file is made automatically for this dataset. Convert it to dataset.json.gz (see the top of experiments/download_exfoliation_data.py) before training.
