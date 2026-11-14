# LM26: 26 RPA interlayer binding energies (Bjorkman et al.)

- Folder: `data/exfoliation/lm26_bjorkman/`
- Source: Tables in https://arxiv.org/abs/1206.3542 and Table II of https://arxiv.org/abs/1406.2436
- Paper: Bjorkman et al., Phys. Rev. Lett. 108, 235502 (2012)
- Licence: values from the papers
- Download method: manual
- Has exfoliation labels in a common format: no (raw files only)

## Notes
No downloadable file. Copy the 26 values from the paper tables into raw/lm26_template.csv (columns: formula,binding_energy_meV_per_A2,source_table). They are per area, not per atom.

## Train and test QFA-Net on it

No common-format file is made automatically for this dataset. Convert it to dataset.json.gz (see the top of experiments/download_exfoliation_data.py) before training.
