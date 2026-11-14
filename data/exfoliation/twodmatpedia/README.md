# 2DMatPedia (6,351 monolayers; exfoliation_energy_per_atom for top-down entries)

- Folder: `data/exfoliation/twodmatpedia/`
- Source: http://www.2dmatpedia.org/download (mirror: https://colabfit.org/id/DS_hdv6si8yu2mv_0, DOI https://doi.org/10.60732/a2df077f)
- Paper: Zhou et al., Sci. Data 6, 86 (2019), https://doi.org/10.1038/s41597-019-0097-3
- Licence: CC-BY 4.0
- Download method: web
- Has exfoliation labels in a common format: yes -> dataset.json.gz

## Notes
PBE without vdW for exfoliation, so values differ from JARVIS (OptB88vdW). The structure stored is the monolayer. exfoliation_energy_per_atom is in eV/atom and converted to meV/atom.

## Train and test QFA-Net on it

```
python experiments/run_qfa_exfoliation.py --dataset twodmatpedia --folds 0
```
