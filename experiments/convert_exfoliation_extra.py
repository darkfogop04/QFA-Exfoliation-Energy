"""
Convert exfoliation datasets that download_exfoliation_data.py keeps as "raw only" into the common format
(data/exfoliation/<dataset>/dataset.json.gz), when their raw files contain per-material labels.

    python experiments/convert_exfoliation_extra.py --only massive_discovery_2025

Common format: JSON list of {"id", "formula", "structure" (pymatgen dict), "target"}; summary.json states the unit.
"""
import argparse
import datetime
import gzip
import json
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "exfoliation"))


def _arr(x):
    """ASE-db JSON array {"__ndarray__": [shape, dtype, flat]} -> nested list."""
    import numpy as np
    if isinstance(x, dict) and "array" in x:
        x = x["array"]
    if isinstance(x, dict) and "__ndarray__" in x:
        shape, _, flat = x["__ndarray__"]
        return np.array(flat).reshape(shape)
    return np.array(x)


def save(name, entries, unit, extra):
    d = os.path.join(ROOT, name)
    with gzip.open(os.path.join(d, "dataset.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(entries, f)
    t = [e["target"] for e in entries]
    json.dump({"dataset": name, "n_labelled": len(entries), "unit": unit, "target_min": min(t), "target_max": max(t),
               "target_mean": sum(t) / len(t), "converted": datetime.date.today().isoformat(), **extra},
              open(os.path.join(d, "summary.json"), "w"), indent=1)
    print(f"{name}: wrote dataset.json.gz with {len(entries)} entries ({unit}), "
          f"target {min(t):.2f} to {max(t):.2f}")


def massive_discovery_2025():
    from pymatgen.core import Lattice, Structure
    name = "massive_discovery_2025"
    z = zipfile.ZipFile(os.path.join(ROOT, name, "raw", "databases.zip"))
    db = {k: v for k, v in json.loads(z.read("2D_materials.json")).items() if isinstance(v, dict)}  # drop ids/nextid
    out, skipped = [], 0
    for key, r in db.items():
        kv = r.get("key_value_pairs", {})
        y = kv.get("E_exf")
        try:
            y = float(y)
            s = Structure(Lattice(_arr(r["cell"])), _arr(r["numbers"]).astype(int).tolist(), _arr(r["positions"]),
                          coords_are_cartesian=True)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if y != y:
            skipped += 1
            continue
        out.append({"id": kv.get("mpid") or f"md-{key}", "formula": s.composition.reduced_formula,
                    "structure": s.as_dict(), "target": y})
    save(name, out, "meV/A^2 (interlayer binding energy, MatterSim UMLIP)",
         {"label_field": "E_exf (meV/A^2)", "n_entries_total": len(db), "skipped": skipped,
          "source": "https://doi.org/10.5281/zenodo.17035156 (CC-BY 4.0)",
          "note": "Labels from the MatterSim universal ML potential, not DFT; per area, not per atom; bulk structures"})


def mc2d_mounet2018():
    """EE_and_PE_structures.txt: one row per (2D layer, 3D parent) with binding energies (meV/A^2, DF2-C09 and
    rVV10) and the file of the exfoliated 2D structure. Target = DF2-C09; one entry per 2D structure file
    (first row kept when several 3D parents give the same layer)."""
    import tarfile
    from pymatgen.core import Lattice, Structure
    name = "mc2d_mounet2018"
    tf = tarfile.open(os.path.join(ROOT, name, "raw", "2D_materials.tar.gz"))
    members = {m.name: m for m in tf.getmembers()}
    rows = tf.extractfile(members["EE_and_PE_structures.txt"]).read().decode().splitlines()
    head = rows[0].split()
    i_be, i_fn, i_ee = head.index("BE_DF2C09[meV/A^2]"), head.index("FILENAME"), head.index("EE/PE")
    out, seen, skipped = [], set(), 0
    for line in rows[1:]:
        c = line.split()
        if len(c) < len(head):
            skipped += 1
            continue
        fn = c[i_fn] + ".json"
        try:
            y = float(c[i_be])
            r = json.loads(tf.extractfile(members[fn]).read())
            r = r[str(r["ids"][0])]
            s = Structure(Lattice(r["cell"]), r["numbers"], r["positions"], coords_are_cartesian=True)
        except Exception:  # noqa: BLE001  missing value "-" or missing file
            skipped += 1
            continue
        if fn in seen:
            continue
        seen.add(fn)
        out.append({"id": os.path.basename(c[i_fn]), "formula": s.composition.reduced_formula,
                    "structure": s.as_dict(), "target": y, "class": c[i_ee]})
    save(name, out, "meV/A^2 (interlayer binding energy, DF2-C09)",
         {"label_field": "BE_DF2C09[meV/A^2] from EE_and_PE_structures.txt", "n_rows": len(rows) - 1,
          "skipped": skipped, "source": "https://archive.materialscloud.org/records/rbqe0-q3717 (CC-BY 4.0)",
          "note": "Monolayer (exfoliated 2D) structures; per area, not per atom; easily (EE) and potentially (PE) "
                  "exfoliable materials only"})


def mc2d_campi2022():
    """structure_2d.json: per 2D material, DF2-C09 binding energy (eV/A^2, lowest over 3D parents) and the uuid of
    the as-extracted (unrelaxed) 2D structure, on which the binding energy was computed. Structures from
    as_extracted_2d_structures.zip (.xsf, else .cif). The 9 GB AiiDA export is not needed."""
    from pymatgen.io.cif import CifParser
    from pymatgen.io.xcrysden import XSF
    name = "mc2d_campi2022"
    raw = os.path.join(ROOT, name, "raw")
    recs = json.load(open(os.path.join(raw, "structure_2d.json")))
    z = zipfile.ZipFile(os.path.join(raw, "as_extracted_2d_structures.zip"))
    files = {os.path.basename(n): n for n in z.namelist()}
    out, skipped = [], 0
    for r in recs:
        be, uid = r.get("binding_energy_per_substructure_per_unit_area_df2"), r.get("as_extracted_2D_structure_uuid")
        try:
            y = 1000.0 * float(be["value"])
            if f"{uid}.xsf" in files:
                s = XSF.from_str(z.read(files[f"{uid}.xsf"]).decode()).structure
            else:
                s = CifParser.from_str(z.read(files[f"{uid}.cif"]).decode()).parse_structures(primitive=False)[0]
        except Exception:  # noqa: BLE001  no DF2 value or no structure file
            skipped += 1
            continue
        out.append({"id": uid, "formula": s.composition.reduced_formula, "structure": s.as_dict(), "target": y})
    save(name, out, "meV/A^2 (interlayer binding energy, DF2-C09)",
         {"label_field": "binding_energy_per_substructure_per_unit_area_df2 (eV/A^2 x 1000)", "n_entries_total": len(recs),
          "skipped": skipped, "source": "https://doi.org/10.24435/materialscloud:36-nd (CC-BY 4.0)",
          "note": "As-extracted (unrelaxed) monolayers, as used for the binding energy; per area; expanded MC2D, "
                  "overlaps mc2d_mounet2018"})


CONVERTERS = {"massive_discovery_2025": massive_discovery_2025, "mc2d_mounet2018": mc2d_mounet2018,
              "mc2d_campi2022": mc2d_campi2022}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(CONVERTERS))
    a = ap.parse_args()
    for n in a.only.split(","):
        CONVERTERS[n]()
