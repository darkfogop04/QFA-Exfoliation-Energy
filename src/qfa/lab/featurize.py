"""
Composition (Magpie) and symmetry features, in pure NumPy.

This re-implements matminer's ElementProperty "magpie" preset without pymatgen:
22 element properties x 6 statistics (min, max, range, mean, avg_dev, mode),
weighted by atomic fraction = 132 numbers. Element tables are matminer's own
files (magpie_data/, BSD licence). Missing values are imputed with the column mean
over all elements, as matminer does with impute_nan=True.

Symmetry features (only for tasks that have a structure): crystal-system one-hot (7)
and space-group number / 230 (1) = 8 numbers.
"""
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

SYMBOLS: List[str] = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}

MAGPIE_PROPS: List[str] = [
    "Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row", "CovalentRadius",
    "Electronegativity", "NsValence", "NpValence", "NdValence", "NfValence", "NValence", "NsUnfilled",
    "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom",
    "SpaceGroupNumber",
]
STATS: List[str] = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"]
CRYSTAL_SYSTEMS: List[str] = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal",
                              "hexagonal", "cubic"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "magpie_data")
_TABLE_CACHE: Dict[str, np.ndarray] = {}


def load_table(name: str) -> np.ndarray:
    """Property values indexed by Z-1, NaN-imputed with the mean over elements."""
    if name not in _TABLE_CACHE:
        vals = []
        with open(os.path.join(_DATA_DIR, name + ".table"), encoding="utf-8") as fh:
            for line in fh:
                tok = line.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok.split()[0]))
                except ValueError:  # "Missing"
                    vals.append(np.nan)
        arr = np.array(vals, dtype=float)
        arr[np.isnan(arr)] = np.nanmean(arr)
        _TABLE_CACHE[name] = arr
    return _TABLE_CACHE[name]


_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def _is_num(tok: str) -> bool:
    return tok[0].isdigit() or tok[0] == "."


def parse_formula(formula: str) -> Dict[str, float]:
    """'Hf2 Si2 Te2', 'Ag(AuS)2', 'Fe0.62C0.00095...' -> {element: amount}."""
    tokens = _TOKEN.findall(formula.replace(" ", ""))
    stack: List[Dict[str, float]] = [{}]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "(":
            stack.append({})
            i += 1
            continue
        if tok == ")":
            group = stack.pop()
            mult = 1.0
            if i + 1 < len(tokens) and _is_num(tokens[i + 1]):
                mult = float(tokens[i + 1])
                i += 1
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + n * mult
            i += 1
            continue
        if tok[0].isalpha():
            if tok not in Z_OF:
                raise ValueError(f"unknown element {tok!r} in {formula!r}")
            amt = 1.0
            if i + 1 < len(tokens) and _is_num(tokens[i + 1]):
                amt = float(tokens[i + 1])
                i += 1
            stack[-1][tok] = stack[-1].get(tok, 0.0) + amt
            i += 1
            continue
        raise ValueError(f"cannot parse {formula!r} at {tok!r}")
    if len(stack) != 1:
        raise ValueError(f"unbalanced parentheses in {formula!r}")
    return {el: n for el, n in stack[0].items() if n > 0}


def magpie_features(formula: str) -> np.ndarray:
    comp = parse_formula(formula)
    els = list(comp)
    w = np.array([comp[e] for e in els], dtype=float)
    w = w / w.sum()
    zi = np.array([Z_OF[e] - 1 for e in els])
    out = []
    for prop in MAGPIE_PROPS:
        v = load_table(prop)[zi]
        mean = float(np.dot(w, v))
        most = np.isclose(w, w.max())
        out += [v.min(), v.max(), v.max() - v.min(), mean, float(np.dot(w, np.abs(v - mean))), v[most].min()]
    return np.array(out, dtype=float)


def magpie_names() -> List[str]:
    return [f"MagpieData {s} {p}" for p in MAGPIE_PROPS for s in STATS]


def symmetry_features(spg_num: int, crys_sys: str) -> np.ndarray:
    one_hot = np.zeros(len(CRYSTAL_SYSTEMS))
    one_hot[CRYSTAL_SYSTEMS.index(crys_sys.lower())] = 1.0
    return np.concatenate([one_hot, [float(spg_num) / 230.0]])


def symmetry_names() -> List[str]:
    return [f"crystal_system={c}" for c in CRYSTAL_SYSTEMS] + ["spacegroup/230"]


def featurize(formulas: Sequence[str], spg: Sequence[int] = None,
              crys: Sequence[str] = None) -> Tuple[np.ndarray, List[str]]:
    X = np.stack([magpie_features(f) for f in formulas])
    names = magpie_names()
    if spg is not None:
        S = np.stack([symmetry_features(s, c) for s, c in zip(spg, crys)])
        X = np.concatenate([X, S], axis=1)
        names = names + symmetry_names()
    return X, names
