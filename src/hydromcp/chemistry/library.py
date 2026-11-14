"""Model compounds and reaction pathways common in hydroprocessing studies."""

from __future__ import annotations

MODEL_COMPOUNDS: dict[str, dict] = {
    # sulfur
    "thiophene": {"smiles": "c1ccsc1", "class": "sulfur", "notes": "Naphtha-range, easy HDS"},
    "benzothiophene": {"smiles": "c1ccc2sccc2c1", "class": "sulfur"},
    "dibenzothiophene": {"smiles": "c1ccc2c(c1)sc1ccccc12", "class": "sulfur",
                         "notes": "Reference DBT; mostly direct desulfurization on CoMo"},
    "4-methyldibenzothiophene": {"smiles": "Cc1cccc2c1sc1ccccc12", "class": "sulfur"},
    "4,6-dimethyldibenzothiophene": {"smiles": "Cc1cccc2c1sc1c(C)cccc12", "class": "sulfur",
                                     "notes": "Refractory in ULSD; hydrogenation route favoured (NiMo)"},
    "ethanethiol": {"smiles": "CCS", "class": "sulfur"},
    "diethyl sulfide": {"smiles": "CCSCC", "class": "sulfur"},
    # nitrogen
    "pyridine": {"smiles": "c1ccncc1", "class": "nitrogen", "notes": "basic"},
    "quinoline": {"smiles": "c1ccc2ncccc2c1", "class": "nitrogen", "notes": "basic, HDS inhibitor"},
    "acridine": {"smiles": "c1ccc2nc3ccccc3cc2c1", "class": "nitrogen", "notes": "basic"},
    "indole": {"smiles": "c1ccc2[nH]ccc2c1", "class": "nitrogen", "notes": "non-basic"},
    "carbazole": {"smiles": "c1ccc2c(c1)[nH]c1ccccc12", "class": "nitrogen", "notes": "non-basic"},
    # aromatics
    "benzene": {"smiles": "c1ccccc1", "class": "aromatic"},
    "toluene": {"smiles": "Cc1ccccc1", "class": "aromatic"},
    "naphthalene": {"smiles": "c1ccc2ccccc2c1", "class": "aromatic"},
    "tetralin": {"smiles": "c1ccc2c(c1)CCCC2", "class": "aromatic"},
    "phenanthrene": {"smiles": "c1ccc2c(c1)ccc1ccccc12", "class": "aromatic"},
    "pyrene": {"smiles": "c1cc2ccc3cccc4ccc(c1)c2c34", "class": "aromatic"},
    # saturates / products
    "biphenyl": {"smiles": "c1ccc(cc1)-c1ccccc1", "class": "product"},
    "cyclohexylbenzene": {"smiles": "C1CCC(CC1)c1ccccc1", "class": "product"},
    "bicyclohexyl": {"smiles": "C1CCC(CC1)C1CCCCC1", "class": "product"},
    "3,3'-dimethylbiphenyl": {"smiles": "Cc1cccc(c1)-c1cccc(C)c1", "class": "product"},
    "propylbenzene": {"smiles": "CCCc1ccccc1", "class": "product"},
    "propylcyclohexane": {"smiles": "CCCC1CCCCC1", "class": "product"},
    "decalin": {"smiles": "C1CCC2CCCCC2C1", "class": "product"},
    "n-butane": {"smiles": "CCCC", "class": "product"},
    "ethane": {"smiles": "CC", "class": "product"},
    "n-hexadecane": {"smiles": "CCCCCCCCCCCCCCCC", "class": "paraffin"},
    "h2s": {"smiles": "S", "class": "gas"},
    "nh3": {"smiles": "N", "class": "gas"},
    "hydrogen": {"smiles": "[H][H]", "class": "gas"},
}

# pathway -> (reactant, [products]); H2 demand is computed from the atom balance
PATHWAYS: dict[str, dict] = {
    "thiophene_hds": {"reactant": "thiophene", "products": ["n-butane", "h2s"]},
    "dbt_dds": {"reactant": "dibenzothiophene", "products": ["biphenyl", "h2s"],
                "notes": "Direct desulfurization (hydrogenolysis)"},
    "dbt_hyd": {"reactant": "dibenzothiophene", "products": ["cyclohexylbenzene", "h2s"],
                "notes": "Hydrogenation route"},
    "dmdbt_dds": {"reactant": "4,6-dimethyldibenzothiophene",
                  "products": ["3,3'-dimethylbiphenyl", "h2s"]},
    "dmdbt_hyd": {"reactant": "4,6-dimethyldibenzothiophene",
                  "products": ["Cc1cccc(c1)C1CCCC(C)C1", "h2s"]},
    "quinoline_hdn": {"reactant": "quinoline", "products": ["propylcyclohexane", "nh3"]},
    "quinoline_hdn_aromatic": {"reactant": "quinoline", "products": ["propylbenzene", "nh3"]},
    "carbazole_hdn": {"reactant": "carbazole", "products": ["bicyclohexyl", "nh3"]},
    "naphthalene_to_tetralin": {"reactant": "naphthalene", "products": ["tetralin"]},
    "naphthalene_to_decalin": {"reactant": "naphthalene", "products": ["decalin"]},
    "benzene_saturation": {"reactant": "benzene", "products": ["C1CCCCC1"]},
}


def lookup(name_or_class: str | None = None) -> list[dict]:
    key = (name_or_class or "").lower()
    out = []
    for name, d in MODEL_COMPOUNDS.items():
        if not key or key in name or key == d["class"]:
            out.append({"name": name, **d})
    return out
