"""
Download every public exfoliation-energy dataset into its own folder, and convert the labelled ones into one
common format that experiments/run_qfa_exfoliation.py can train and test on.

Run on your own machine (needs internet), inside the project venv:

    pip install -U jarvis-tools matminer requests
    python experiments/download_exfoliation_data.py                 # everything except the 9 GB MC2D export
    python experiments/download_exfoliation_data.py --list          # show the datasets and exit
    python experiments/download_exfoliation_data.py --only jarvis_dft_3d,twodmatpedia
    python experiments/download_exfoliation_data.py --include-huge  # also the ~9 GB Campi 2022 AiiDA export
    python experiments/download_exfoliation_data.py --readme-only   # only create the folders and README files

Layout (one folder per dataset):

    data/exfoliation/<dataset>/README.md          what it is, source, licence, fields
    data/exfoliation/<dataset>/raw/...            the files exactly as downloaded
    data/exfoliation/<dataset>/dataset.json.gz    (labelled sets only) common format, see below
    data/exfoliation/<dataset>/summary.json       entry count, target range, download date
    data/exfoliation/STATUS.md                    what was downloaded, skipped or failed

Common format (dataset.json.gz): a JSON list, one entry per material:
    {"id": str, "formula": str, "structure": <pymatgen Structure.as_dict()>, "target": float (meV/atom)}

Re-running skips files that are already there, so an interrupted download can be resumed.
"""
import argparse
import datetime
import gzip
import json
import os
import re
import shutil
import sys
import time
import zipfile
from typing import Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "exfoliation"))
UA = {"User-Agent": "QFA-Net-MTP-dataset-downloader/1.0 (research use)"}

# ----------------------------------------------------------------------------------------------------------------
# Dataset catalogue. kind: "jarvis" (jarvis-tools), "matminer", "web" (2DMatPedia page), "mcloud" (Materials Cloud
# archive record), "manual" (no public direct download: README with instructions only).
# ----------------------------------------------------------------------------------------------------------------
DATASETS: Dict[str, Dict] = {
    "jarvis_dft_3d": {
        "kind": "jarvis", "jarvis_name": "dft_3d", "labelled": True,
        "title": "JARVIS-DFT 3D (bulk crystals; exfoliation energy where computed)",
        "source": "https://pages.nist.gov/jarvis/databases/ (Figshare, via jarvis-tools)",
        "paper": "Choudhary et al., Sci. Rep. 7, 5179 (2017), https://doi.org/10.1038/s41598-017-05402-0; "
                 "Choudhary et al., npj Comput. Mater. 6, 173 (2020)",
        "licence": "NIST public data (see the JARVIS terms of use)",
        "notes": "Bulk structures. Only entries with a numeric 'exfoliation_energy' (meV/atom, OptB88vdW) go into "
                 "dataset.json.gz. matbench_jdft2d is a cleaned subset of these.",
    },
    "jarvis_dft_2d": {
        "kind": "jarvis", "jarvis_name": "dft_2d", "labelled": True,
        "title": "JARVIS-DFT 2D (monolayers)",
        "source": "https://pages.nist.gov/jarvis/databases/ (Figshare, via jarvis-tools)",
        "paper": "Choudhary et al., Sci. Rep. 7, 5179 (2017)",
        "licence": "NIST public data (see the JARVIS terms of use)",
        "notes": "Monolayer structures. If an 'exfoliation_energy' field is present it is used; otherwise the raw "
                 "file is kept and dataset.json.gz is not written (the labels then live in jarvis_dft_3d).",
    },
    "matbench_jdft2d": {
        "kind": "matminer", "matminer_name": "matbench_jdft2d", "labelled": True,
        "title": "MatBench jdft2d (636 bulk crystals, exfoliation energy)",
        "source": "https://hackingmaterials.lbl.gov/matminer/dataset_summary.html (matminer.datasets)",
        "paper": "Dunn et al., npj Comput. Mater. 6, 138 (2020), https://doi.org/10.1038/s41524-020-00406-3",
        "licence": "CC-BY 4.0 (MatBench)",
        "notes": "The benchmark version. Train/test with the official 5 folds (data/matbench_meta/folds.json); "
                 "run_qfa_exfoliation.py does this automatically for this folder.",
    },
    "twodmatpedia": {
        "kind": "web", "page": "http://www.2dmatpedia.org/download", "labelled": True,
        "title": "2DMatPedia (6,351 monolayers; exfoliation_energy_per_atom for top-down entries)",
        "source": "http://www.2dmatpedia.org/download (mirror: https://colabfit.org/id/DS_hdv6si8yu2mv_0, "
                  "DOI https://doi.org/10.60732/a2df077f)",
        "paper": "Zhou et al., Sci. Data 6, 86 (2019), https://doi.org/10.1038/s41597-019-0097-3",
        "licence": "CC-BY 4.0",
        "notes": "PBE without vdW for exfoliation, so values differ from JARVIS (OptB88vdW). The structure stored is "
                 "the monolayer. exfoliation_energy_per_atom is in eV/atom and converted to meV/atom.",
    },
    "mc2d_mounet2018": {
        "kind": "mcloud", "record": "rbqe0-q3717", "doi": "10.24435/materialscloud:2017.0008/v3",
        "labelled": False, "huge": False,
        "title": "Materials Cloud 2D, Mounet et al. 2018 (~1,825 easily exfoliable compounds)",
        "source": "https://archive.materialscloud.org/records/rbqe0-q3717 (~953 MiB)",
        "paper": "Mounet et al., Nat. Nanotechnol. 13, 246 (2018), https://doi.org/10.1038/s41565-017-0035-5",
        "licence": "CC-BY 4.0 (see the record)",
        "notes": "Binding energies (meV/A^2, DF2-C09 and rVV10) rather than meV/atom; the record has simple tables "
                 "plus AiiDA exports. Raw files only; inspect the tables before training (see README in raw/).",
    },
    "mc2d_campi2022": {
        "kind": "mcloud", "doi": "10.24435/materialscloud:36-nd", "labelled": False, "huge": True,
        "title": "Materials Cloud 2D, Campi et al. 2022 (expanded MC2D)",
        "source": "https://doi.org/10.24435/materialscloud:36-nd (main AiiDA export ~9 GB)",
        "paper": "Campi et al., ACS Nano 17, 11268 (2023)",
        "licence": "CC-BY 4.0 (see the record)",
        "notes": "Files above 1 GB are skipped unless you pass --include-huge. Reading the AiiDA export needs AiiDA.",
    },
    "vahdat_exfoliability": {
        "kind": "mcloud", "record_old": "1378", "doi": "10.24435/materialscloud:m4-7f", "labelled": False,
        "huge": False,
        "title": "Vahdat et al., machine-learning exfoliability dataset",
        "source": "https://archive.materialscloud.org/record/1378 (zip ~281.5 MiB)",
        "paper": "see the Materials Cloud record (DOI 10.24435/materialscloud:m4-7f)",
        "licence": "see the record",
        "notes": "Everything needed to replicate their model. The zip is unpacked into raw/unzipped/.",
    },
    "alexandria_2d": {
        "kind": "jarvis", "jarvis_name": "alex_pbe_2d_all", "labelled": False,
        "title": "Alexandria 2D (~200k PBE 2D structures, no exfoliation labels)",
        "source": "https://pages.nist.gov/jarvis/databases/ (alex_pbe_2d_all, via jarvis-tools)",
        "paper": "Schmidt et al., Alexandria database",
        "licence": "CC-BY 4.0",
        "notes": "No exfoliation-energy labels: useful for pretraining or screening, not for testing. Large.",
    },
    "c2db": {
        "kind": "manual", "labelled": False,
        "title": "C2DB, Computational 2D Materials Database (DTU)",
        "source": "https://c2db.fysik.dtu.dk ; downloads listed on the DTU CMR C2DB page (c2db.db, c2db.tar.gz)",
        "paper": "Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et al., 2D Mater. 8, 044002 (2021)",
        "licence": "see the DTU CMR terms",
        "notes": "The full database is given on request. Open the CMR C2DB page, download c2db.db (ASE database) "
                 "into raw/, then read it with `ase db raw/c2db.db`.",
    },
    "ashton_materialsweb": {
        "kind": "manual", "labelled": False,
        "title": "Ashton et al. 2017 (826 monolayers, MaterialsWeb)",
        "source": "https://materialsweb.org (could not be confirmed as live)",
        "paper": "Ashton et al., Phys. Rev. Lett. 118, 106101 (2017), https://arxiv.org/abs/1610.07673",
        "licence": "see the site",
        "notes": "If the site works, save the structures into raw/. Otherwise use the paper's supplement.",
    },
    "lm26_bjorkman": {
        "kind": "manual", "labelled": False,
        "title": "LM26: 26 RPA interlayer binding energies (Bjorkman et al.)",
        "source": "Tables in https://arxiv.org/abs/1206.3542 and Table II of https://arxiv.org/abs/1406.2436",
        "paper": "Bjorkman et al., Phys. Rev. Lett. 108, 235502 (2012)",
        "licence": "values from the papers",
        "notes": "No downloadable file. Copy the 26 values from the paper tables into raw/lm26_template.csv "
                 "(columns: formula,binding_energy_meV_per_A2,source_table). They are per area, not per atom.",
    },
    "no_public_data": {
        "kind": "manual", "labelled": False,
        "title": "Papers without a data link found (Massive Discovery 2025, X2DB, ML2DDB)",
        "source": "https://arxiv.org/pdf/2509.20164 ; https://pubmed.ncbi.nlm.nih.gov/41945467/ ; "
                  "https://arxiv.org/pdf/2507.00584",
        "paper": "see links",
        "licence": "-",
        "notes": "Check each paper's 'Data availability' section; put any files you obtain into raw/.",
    },
}

HUGE_BYTES = 1_000_000_000


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def folder(name: str) -> str:
    d = os.path.join(ROOT, name)
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    return d


def write_readme(name: str) -> None:
    m = DATASETS[name]
    d = folder(name)
    lines = [f"# {m['title']}", "",
             f"- Folder: `data/exfoliation/{name}/`",
             f"- Source: {m['source']}",
             f"- Paper: {m['paper']}",
             f"- Licence: {m['licence']}",
             f"- Download method: {m['kind']}",
             f"- Has exfoliation labels in a common format: {'yes -> dataset.json.gz' if m['labelled'] else 'no (raw files only)'}",
             "", "## Notes", m["notes"], "",
             "## Train and test QFA-Net on it", ""]
    if m["labelled"]:
        lines += ["```", f"python experiments/run_qfa_exfoliation.py --dataset {name} --folds 0", "```"]
    else:
        lines += ["No common-format file is made automatically for this dataset. Convert it to dataset.json.gz "
                  "(see the top of experiments/download_exfoliation_data.py) before training."]
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if name == "lm26_bjorkman":
        p = os.path.join(d, "raw", "lm26_template.csv")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("formula,binding_energy_meV_per_A2,source_table\n")


def http_get(url: str, stream: bool = False, timeout: int = 60):
    import requests
    r = requests.get(url, headers=UA, stream=stream, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def download(url: str, path: str, expected: Optional[int] = None) -> str:
    if os.path.exists(path) and (expected is None or os.path.getsize(path) == expected):
        print(f"    have {os.path.basename(path)}")
        return path
    tmp = path + ".part"
    t0 = time.time()
    with http_get(url, stream=True, timeout=120) as r:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {os.path.basename(path)}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="")
    os.replace(tmp, path)
    print(f"\r    saved {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")
    return path


def save_common(name: str, entries: List[Dict], extra: Dict) -> None:
    d = folder(name)
    entries = [e for e in entries if e.get("structure") and isinstance(e.get("target"), (int, float))]
    with gzip.open(os.path.join(d, "dataset.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(entries, f)
    t = [e["target"] for e in entries]
    summ = {"dataset": name, "n_labelled": len(entries), "unit": "meV/atom",
            "target_min": min(t) if t else None, "target_max": max(t) if t else None,
            "target_mean": sum(t) / len(t) if t else None,
            "downloaded": datetime.date.today().isoformat(), **extra}
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=1)
    print(f"    wrote dataset.json.gz with {len(entries)} labelled entries")


def as_float(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


# ----------------------------------------------------------------------------------------------------------------
# per-kind download functions
# ----------------------------------------------------------------------------------------------------------------
def get_jarvis(name: str, m: Dict) -> str:
    from jarvis.db.figshare import data
    d = folder(name)
    rows = data(m["jarvis_name"], store_dir=os.path.join(d, "raw"))   # jarvis-tools keeps its zip in raw/
    keys = sorted({k for r in rows[:200] for k in r})
    exf_keys = [k for k in keys if "exfol" in k.lower()]
    print(f"    {len(rows)} entries; exfoliation field(s): {exf_keys or 'none'}")
    if not (m["labelled"] and exf_keys):
        json.dump({"dataset": name, "n_entries": len(rows), "fields": keys,
                   "downloaded": datetime.date.today().isoformat()},
                  open(os.path.join(d, "summary.json"), "w"), indent=1)
        return f"{len(rows)} entries (raw only)"
    from jarvis.core.atoms import Atoms
    key = exf_keys[0]
    out = []
    for r in rows:
        y = as_float(r.get(key))
        if y is None:
            continue
        try:
            s = Atoms.from_dict(r["atoms"]).pymatgen_converter()
        except Exception:
            continue
        out.append({"id": str(r.get("jid", len(out))), "formula": s.composition.reduced_formula,
                    "structure": s.as_dict(), "target": y})
    save_common(name, out, {"label_field": key, "n_entries_total": len(rows)})
    return f"{len(out)} labelled of {len(rows)}"


def get_matminer(name: str, m: Dict) -> str:
    from matminer.datasets import load_dataset
    d = folder(name)
    df = load_dataset(m["matminer_name"], data_home=os.path.join(d, "raw"))
    tcol = [c for c in df.columns if c != "structure"][0]
    out = [{"id": f"{m['matminer_name']}-{i + 1}", "formula": s.composition.reduced_formula,
            "structure": s.as_dict(), "target": float(y)} for i, (s, y) in enumerate(zip(df["structure"], df[tcol]))]
    save_common(name, out, {"label_field": tcol, "official_folds": "data/matbench_meta/folds.json"})
    return f"{len(out)} labelled"


def get_2dmatpedia(name: str, m: Dict) -> str:
    d = folder(name)
    page = http_get(m["page"]).text
    links = sorted(set(re.findall(r'href=["\']([^"\']+\.(?:json(?:\.gz)?|gz|zip))["\']', page, re.I)))
    links = [ln for ln in links if "band" not in ln.lower()]   # skip the 3.9 GB band-structure tarball (no labels)
    if not links:
        raise RuntimeError(f"no data link found on {m['page']}; download the JSON by hand into {d}\\raw")
    files = []
    for ln in links:
        url = ln if ln.startswith("http") else re.sub(r"/download/?$", "", m["page"]) + "/" + ln.lstrip("/")
        files.append(download(url, os.path.join(d, "raw", os.path.basename(ln.split("?")[0]))))
    recs = []
    for p in files:
        recs += read_json_any(p)
    from pymatgen.core import Structure
    out = []
    for r in recs:
        y = as_float(r.get("exfoliation_energy_per_atom"))
        if y is None or not r.get("structure"):
            continue
        try:
            s = Structure.from_dict(r["structure"])
        except Exception:
            continue
        out.append({"id": str(r.get("material_id", len(out))), "formula": s.composition.reduced_formula,
                    "structure": s.as_dict(), "target": 1000.0 * y})
    save_common(name, out, {"label_field": "exfoliation_energy_per_atom (eV/atom x 1000)",
                            "n_entries_total": len(recs)})
    return f"{len(out)} labelled of {len(recs)}"


def read_json_any(path: str) -> List[Dict]:
    """JSON list, JSON object, or one JSON object per line; plain, .gz or .zip."""
    if path.endswith(".zip"):
        out = []
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith((".json", ".json.gz")):
                    tmp = os.path.join(os.path.dirname(path), os.path.basename(n))
                    with z.open(n) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    out += read_json_any(tmp)
        return out
    with open(path, "rb") as fb:   # requests may already have un-gzipped a ".gz" download: check the magic bytes
        op = gzip.open if fb.read(2) == b"\x1f\x8b" else open
    with op(path, "rt", encoding="utf-8") as f:
        text = f.read()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


def mcloud_record_id(m: Dict) -> str:
    if m.get("record"):
        return m["record"]
    for url in ([f"https://archive.materialscloud.org/record/{m['record_old']}"] if m.get("record_old") else []) + \
               [f"https://doi.org/{m['doi']}"]:
        try:
            final = http_get(url).url
        except Exception:
            continue
        hit = re.search(r"/records?/([A-Za-z0-9\-]+)", final)
        if hit:
            return hit.group(1)
    raise RuntimeError(f"could not resolve the Materials Cloud record for DOI {m['doi']}")


def get_mcloud(name: str, m: Dict, include_huge: bool) -> str:
    d = folder(name)
    rid = mcloud_record_id(m)
    api = f"https://archive.materialscloud.org/api/records/{rid}/files"
    entries = http_get(api).json().get("entries", [])
    if not entries:
        raise RuntimeError(f"no files listed at {api}")
    got, skipped = [], []
    for e in entries:
        key, size = e["key"], int(e.get("size") or 0)
        if size > HUGE_BYTES and not include_huge:
            skipped.append(f"{key} ({size / 1e9:.1f} GB)")
            continue
        url = e.get("links", {}).get("content") or f"https://archive.materialscloud.org/records/{rid}/files/{key}?download=1"
        p = download(url, os.path.join(d, "raw", key), expected=size or None)
        got.append(key)
        if key.endswith(".zip"):
            tgt = os.path.join(d, "raw", "unzipped")
            if not os.path.isdir(tgt):
                with zipfile.ZipFile(p) as z:
                    z.extractall(tgt)
    json.dump({"dataset": name, "record": rid, "downloaded_files": got, "skipped_huge": skipped,
               "downloaded": datetime.date.today().isoformat()},
              open(os.path.join(d, "summary.json"), "w"), indent=1)
    msg = f"{len(got)} file(s)"
    if skipped:
        msg += f"; skipped (use --include-huge): {', '.join(skipped)}"
    return msg


# ----------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated dataset folder names (default: all)")
    ap.add_argument("--skip", default="", help="comma-separated dataset folder names to skip")
    ap.add_argument("--include-huge", action="store_true", help="also download files above 1 GB (MC2D 2022 ~9 GB)")
    ap.add_argument("--readme-only", action="store_true", help="only create the folders and README files")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, m in DATASETS.items():
            print(f"{k:22s} {m['kind']:9s} {'labelled' if m['labelled'] else 'raw only':9s} {m['title']}")
        return
    names = a.only.split(",") if a.only else list(DATASETS)
    names = [n for n in names if n not in set(a.skip.split(","))]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s): {unknown}; see --list")
    os.makedirs(ROOT, exist_ok=True)
    status = []
    for n in names:
        write_readme(n)
        m = DATASETS[n]
        if a.readme_only:
            continue
        print(f"\n== {n}: {m['title']}", flush=True)
        try:
            if m["kind"] == "jarvis":
                msg = get_jarvis(n, m)
            elif m["kind"] == "matminer":
                msg = get_matminer(n, m)
            elif m["kind"] == "web":
                msg = get_2dmatpedia(n, m)
            elif m["kind"] == "mcloud":
                msg = get_mcloud(n, m, a.include_huge)
            else:
                msg = "MANUAL: no direct download; see README.md"
            status.append((n, "ok" if not msg.startswith("MANUAL") else "manual", msg))
        except Exception as e:  # keep going with the other datasets
            status.append((n, "FAILED", f"{type(e).__name__}: {e}"))
            print(f"    FAILED: {type(e).__name__}: {e}")
    if a.readme_only:
        print(f"created {len(names)} folders with README.md under {ROOT}")
        return
    lines = ["# Exfoliation datasets: download status", "",
             f"Run: {datetime.datetime.now().isoformat(timespec='minutes')}", "",
             "| Dataset | Status | Details |", "|---|---|---|"]
    lines += [f"| {n} | {s} | {msg} |" for n, s, msg in status]
    with open(os.path.join(ROOT, "STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
