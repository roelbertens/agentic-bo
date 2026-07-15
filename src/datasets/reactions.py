"""Real reaction-optimisation dataset: Buchwald-Hartwig C-N cross-coupling.

Ahneman, Estrada, Wang, Dreher & Doyle, *Science* 2018 ("Predicting reaction
performance in C-N cross-coupling using machine learning",
doi:10.1126/science.aar5169; commonly the "Dreher-Doyle dataset"): a complete
high-throughput experimentation grid of 3,955 Pd-catalysed reactions combining

    4 ligands  x  3 bases  x  22 isoxazole additives  x  15 aryl halides

each with a measured yield (%). Because it is a full grid, the best possible
yield over the pool is known exactly — no oracle/API needed to score a run.

Two framings are provided:

* ``load_buchwald_hartwig``  — the whole 3,955-reaction pool. Good solutions are
  dense here (~7% of reactions yield >=80%), so even random search reaches ~87%
  of the optimum in 25 tries; the metric saturates and method gaps are small.
* ``load_buchwald_by_substrate`` — the standard *reaction-optimisation* framing:
  fix the aryl halide and optimise the 4x3x22 = 264 ligand/base/additive
  combinations for that substrate, then average over substrates. Smaller pools,
  real structure (many additives poison the reaction), and a low starting
  point — the regime where an agent's cold-start knowledge can pay off.

Why an agent can win here: the search space is *discrete* and made of *named
chemical building blocks*. A GP over one-hot encodings starts blind, whereas an
LLM agent can bring prior chemistry knowledge to bear from round one (e.g. Ramos
et al., ACS Cent. Sci.; Reasoning-BO, arXiv:2505.12833 — full citations in the
README's References section).

Data: Dreher_and_Doyle_input_data.xlsx from
https://github.com/rxn4chemistry/rxn_yields (run scripts/get_data.py).
"""
from __future__ import annotations

import os

import numpy as np

from .base import Dataset

_CATEGORIES = ["Ligand", "Base", "Additive", "Aryl halide"]
_YIELD = "Output"

# Canonical reagents of the Ahneman et al. dataset. Ligands/bases have well-known names; naming
# them (not just the SMILES) is what lets an LLM apply real chemistry knowledge.
# Keyed by the exact SMILES in the dataset; edit freely if you spot a mismatch.
_KNOWN_NAMES = {
    # ligands (the four classic Buchwald dialkylphosphino-biaryls)
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC": "AdBrettPhos",  # noqa: E501
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "tBuBrettPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "tBuXPhos",
    # bases
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et",
    "CN1CCCN2C1=NCCC2": "MTBD",
    "CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG",
}

_PREFIX = {"Ligand": "L", "Base": "B", "Additive": "A", "Aryl halide": "H"}

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "buchwald_hartwig.xlsx")


def _label(value: str) -> str:
    name = _KNOWN_NAMES.get(value)
    return f"{name}  [{value}]" if name else value


def _encode(df, categories, anonymize: bool = False):
    """One-hot encode the given categorical columns; return (X, descriptions, legend).

    With ``anonymize`` the legend lists only the opaque short ids (L1, B2, ...) and
    withholds names/SMILES — the name-ablation probe: if the agent's edge comes from
    named chemistry knowledge, it must disappear here.
    """

    onehot_blocks, legend_lines, short_ids = [], [], {}
    for cat in categories:
        values = list(dict.fromkeys(df[cat].tolist()))  # unique, first-appearance order
        index = {v: i for i, v in enumerate(values)}
        block = np.zeros((len(df), len(values)))
        block[np.arange(len(df)), df[cat].map(index).to_numpy()] = 1.0
        onehot_blocks.append(block)
        if anonymize:
            legend_lines.append(
                f"{cat}s: " + ", ".join(f"{_PREFIX[cat]}{i + 1}" for i in range(len(values))))
        else:
            legend_lines.append(f"{cat}s:")
        for i, v in enumerate(values):
            sid = f"{_PREFIX[cat]}{i + 1}"
            short_ids[(cat, v)] = sid
            if not anonymize:
                legend_lines.append(f"  {sid} = {_label(v)}")

    X = np.hstack(onehot_blocks)
    descriptions = [
        ", ".join(f"{cat.lower().replace(' ', '_')}={short_ids[(cat, row[cat])]}"
                  for cat in categories)
        for _, row in df.iterrows()
    ]
    return X, descriptions, "\n".join(legend_lines)


def _read(path):
    import pandas as pd  # imported lazily; only this dataset needs pandas/openpyxl

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Buchwald-Hartwig data not found at {path}. Run: python scripts/get_data.py"
        )
    # The FullCV_* sheets each contain all 3,955 rows (different CV orderings).
    return pd.read_excel(path, sheet_name="FullCV_01")


def load_buchwald_hartwig(path: str = DEFAULT_PATH, subsample: int | None = None,
                          seed: int = 0, anonymize: bool = False) -> Dataset:
    df = _read(path)
    if subsample is not None and subsample < len(df):
        df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)

    X, descriptions, legend = _encode(df, _CATEGORIES, anonymize=anonymize)
    if anonymize:
        legend = (
            "Each reaction combines one ligand, base, additive and aryl halide, "
            "referenced below by short id. Reagent identities are withheld (name ablation):\n"
            + legend
        )
    else:
        legend = (
            "Each reaction combines one ligand, base, additive and aryl halide, "
            "referenced below by short id. Reagent identities (name and/or SMILES):\n" + legend
        )
    return Dataset(
        X=X, y=df[_YIELD].to_numpy(dtype=float), descriptions=descriptions,
        objective_label="reaction yield (%)",
        title="Buchwald-Hartwig C-N coupling (Ahneman 2018)"
              + (" [anonymised]" if anonymize else ""),
        legend=legend,
    )


DEFAULT_DA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "direct_arylation")


def load_direct_arylation(dir_path: str = DEFAULT_DA_DIR, anonymize: bool = False) -> Dataset:
    """Direct C-H arylation condition optimisation (Shields et al., Nature 2021,
    doi:10.1038/s41586-021-03213-y).

    A full-factorial pool of 1,728 reactions over 12 ligands x 4 bases x 4 solvents
    x 3 concentrations x 3 temperatures, with measured yield (%). This landscape is
    deceptive — a third of the conditions give ~0% yield and the median is ~8% — so
    a GP over encodings starts blind, and it is where an LLM agent's prior knowledge
    (avoid dead ligand/base/solvent combinations) should pay off.
    """
    import pandas as pd

    idx_path = os.path.join(dir_path, "experiment_index.csv")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Direct-arylation data not found at {idx_path}. Run: python scripts/get_data.py"
        )
    df = pd.read_csv(idx_path)

    def name_map(fname, smiles_col, name_col):
        p = os.path.join(dir_path, fname)
        if not os.path.exists(p):
            return {}
        t = pd.read_csv(p)
        return dict(zip(t[smiles_col], t[name_col], strict=True))

    maps = {
        "Ligand_SMILES": name_map("ligand-list.csv", "Ligand_SMILES", "Ligand"),
        "Base_SMILES": name_map("base-list.csv", "Base_SMILES", "Base"),
        "Solvent_SMILES": name_map("solvent-list.csv", "Solvent_SMILES", "Solvent"),
    }
    cat_cols = ["Ligand_SMILES", "Base_SMILES", "Solvent_SMILES"]
    num_cols = ["Concentration", "Temp_C"]

    # One-hot the categoricals (by readable name) + min-max scale the numerics.
    blocks, legend_lines = [], []
    names_per_row = {}
    for col in cat_cols:
        kind = col.split("_")[0]
        values = list(dict.fromkeys(df[col].tolist()))
        index = {v: i for i, v in enumerate(values)}
        if anonymize:
            # Name ablation: opaque ids (L1, B2, S3) instead of reagent names/SMILES.
            ids = {v: f"{kind[0]}{i + 1}" for i, v in enumerate(values)}
            names_per_row[kind] = df[col].map(ids).tolist()
            legend_lines.append(f"{kind}s: " + ", ".join(ids[v] for v in values))
        else:
            m = maps[col]
            names_per_row[kind] = [m.get(v, v) for v in df[col]]
            legend_lines.append(f"{kind}s: " + ", ".join(
                f"{maps[col].get(v, v)} [{v}]" if maps[col] else str(v) for v in values))
        block = np.zeros((len(df), len(values)))
        block[np.arange(len(df)), df[col].map(index).to_numpy()] = 1.0
        blocks.append(block)

    num = df[num_cols].to_numpy(dtype=float)
    lo, hi = num.min(axis=0), num.max(axis=0)
    num_norm = (num - lo) / np.where(hi > lo, hi - lo, 1.0)
    X = np.hstack(blocks + [num_norm])

    descriptions = []
    for i in range(len(df)):
        descriptions.append(
            f"ligand={names_per_row['Ligand'][i]}, base={names_per_row['Base'][i]}, "
            f"solvent={names_per_row['Solvent'][i]}, "
            f"concentration={df['Concentration'].iloc[i]:g} M, temp={df['Temp_C'].iloc[i]:g} C")

    identities = ("Reagent identities are withheld (name ablation); ligands/bases/solvents "
                  "are opaque ids:" if anonymize else "Reagent identities (name [SMILES]):")
    legend = (f"Optimise reaction conditions. {identities}\n"
              + "\n".join(legend_lines)
              + "\nConcentration in M, temperature in C (ranges shown across candidates).")

    return Dataset(
        X=X, y=df["yield"].to_numpy(dtype=float), descriptions=descriptions,
        objective_label="reaction yield (%)",
        title="Direct arylation (Shields 2021)" + (" [anonymised]" if anonymize else ""),
        legend=legend,
    )


def load_buchwald_by_substrate(path: str = DEFAULT_PATH, limit: int | None = None,
                               anonymize: bool = False) -> list:
    """One Dataset per aryl halide: optimise 4x3x22 ligand/base/additive for a fixed substrate.

    Returns up to ``limit`` sub-datasets (substrates ordered by first appearance).
    """
    df = _read(path)
    substrates = list(dict.fromkeys(df["Aryl halide"].tolist()))
    if limit is not None:
        substrates = substrates[:limit]

    datasets = []
    cats = ["Ligand", "Base", "Additive"]  # aryl halide is fixed per sub-dataset
    for i, sub in enumerate(substrates):
        sdf = df[df["Aryl halide"] == sub].reset_index(drop=True)
        X, descriptions, legend = _encode(sdf, cats, anonymize=anonymize)
        sub_label = f"H{i + 1} (identity withheld)" if anonymize else _label(sub)
        identities = ("Reagent identities are withheld (name ablation):" if anonymize
                      else "Reagent identities (name and/or SMILES):")
        legend = (
            f"All reactions below use a FIXED aryl halide: {sub_label}.\n"
            f"Choose the ligand, base and additive. {identities}\n"
            + legend
        )
        datasets.append(Dataset(
            X=X, y=sdf[_YIELD].to_numpy(dtype=float), descriptions=descriptions,
            objective_label="reaction yield (%)",
            title=f"Buchwald-Hartwig, substrate H{i + 1} ({len(sdf)} candidates)"
                  + (" [anonymised]" if anonymize else ""),
            legend=legend,
        ))
    return datasets
