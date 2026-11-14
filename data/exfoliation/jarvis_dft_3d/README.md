# JARVIS-DFT 3D (bulk crystals; exfoliation energy where computed)

- Folder: `data/exfoliation/jarvis_dft_3d/`
- Source: https://pages.nist.gov/jarvis/databases/ (Figshare, via jarvis-tools)
- Paper: Choudhary et al., Sci. Rep. 7, 5179 (2017), https://doi.org/10.1038/s41598-017-05402-0; Choudhary et al., npj Comput. Mater. 6, 173 (2020)
- Licence: NIST public data (see the JARVIS terms of use)
- Download method: jarvis
- Has exfoliation labels in a common format: yes -> dataset.json.gz

## Notes
Bulk structures. Only entries with a numeric 'exfoliation_energy' (meV/atom, OptB88vdW) go into dataset.json.gz. matbench_jdft2d is a cleaned subset of these.

## Train and test QFA-Net on it

```
python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d --folds 0
```
