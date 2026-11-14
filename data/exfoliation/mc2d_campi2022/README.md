# Materials Cloud 2D, Campi et al. 2022 (expanded MC2D)

- Folder: `data/exfoliation/mc2d_campi2022/`
- Source: https://doi.org/10.24435/materialscloud:36-nd (main AiiDA export ~9 GB)
- Paper: Campi et al., ACS Nano 17, 11268 (2023)
- Licence: CC-BY 4.0 (see the record)
- Download method: mcloud
- Has exfoliation labels in a common format: no (raw files only)

## Notes
Files above 1 GB are skipped unless you pass --include-huge. Reading the AiiDA export needs AiiDA.

## Train and test QFA-Net on it

No common-format file is made automatically for this dataset. Convert it to dataset.json.gz (see the top of experiments/download_exfoliation_data.py) before training.
