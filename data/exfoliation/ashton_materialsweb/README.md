# Ashton et al. 2017 (826 monolayers, MaterialsWeb)

- Folder: `data/exfoliation/ashton_materialsweb/`
- Source: https://materialsweb.org (could not be confirmed as live)
- Paper: Ashton et al., Phys. Rev. Lett. 118, 106101 (2017), https://arxiv.org/abs/1610.07673
- Licence: see the site
- Download method: manual
- Has exfoliation labels in a common format: no (raw files only)

## Notes
If the site works, save the structures into raw/. Otherwise use the paper's supplement.

## Train and test QFA-Net on it

No common-format file is made automatically for this dataset. Convert it to dataset.json.gz (see the top of experiments/download_exfoliation_data.py) before training.
