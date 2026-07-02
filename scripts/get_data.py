#!/usr/bin/env python3
"""Download the real reaction datasets (~2 MB) into data/.

Sources (see README "References" for full citations):
* Buchwald-Hartwig: rxn4chemistry/rxn_yields (MIT), from Ahneman, Estrada, Wang,
  Dreher & Doyle, Science 2018, doi:10.1126/science.aar5169.
* Direct arylation: b-shields/edbo (MIT), from Shields et al., Nature 2021,
  doi:10.1038/s41586-021-03213-y.
"""
import os
import urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

FILES = {
    "buchwald_hartwig.xlsx": (
        "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
        "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    ),
    "direct_arylation/experiment_index.csv": None,
    "direct_arylation/ligand-list.csv": None,
    "direct_arylation/base-list.csv": None,
    "direct_arylation/solvent-list.csv": None,
}
_DA_BASE = ("https://raw.githubusercontent.com/b-shields/edbo/master/"
            "examples/DOE/data/direct_arylation/")
for k in list(FILES):
    if FILES[k] is None:
        FILES[k] = _DA_BASE + os.path.basename(k)


def main() -> None:
    for rel, url in FILES.items():
        dest = os.path.join(DATA, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            print(f"Already present: {rel}")
            continue
        print(f"Downloading {rel} ...")
        urllib.request.urlretrieve(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
