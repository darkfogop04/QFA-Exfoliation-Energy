# C2DB, Computational 2D Materials Database (DTU)

- Folder: `data/exfoliation/c2db/`
- Source: https://c2db.fysik.dtu.dk ; downloads listed on the DTU CMR C2DB page (c2db.db, c2db.tar.gz)
- Paper: Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et al., 2D Mater. 8, 044002 (2021)
- Licence: see the DTU CMR terms
- Download method: manual
- Has exfoliation labels in a common format: no (raw files only)

## Notes
The full database is given on request. Open the CMR C2DB page, download c2db.db (ASE database) into raw/, then read it with `ase db raw/c2db.db`.

## Train and test QFA-Net on it

No common-format file is made automatically for this dataset. Convert it to dataset.json.gz (see the top of experiments/download_exfoliation_data.py) before training.
