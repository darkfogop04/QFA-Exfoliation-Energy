# JARVIS-DFT 2D (monolayers)

- Folder: `data/exfoliation/jarvis_dft_2d/`
- Source: https://pages.nist.gov/jarvis/databases/ (Figshare, via jarvis-tools)
- Paper: Choudhary et al., Sci. Rep. 7, 5179 (2017)
- Licence: NIST public data (see the JARVIS terms of use)
- Download method: jarvis
- Has exfoliation labels in a common format: yes -> dataset.json.gz

## Notes
Monolayer structures. If an 'exfoliation_energy' field is present it is used; otherwise the raw file is kept and dataset.json.gz is not written (the labels then live in jarvis_dft_3d).

## Train and test QFA-Net on it

```
python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_2d --folds 0
```
