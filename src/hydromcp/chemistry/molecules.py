"""RDKit molecule utilities for feed/product model compounds."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ATOMIC_MASS = {"C": 12.011, "H": 1.008, "S": 32.06, "N": 14.007, "O": 15.999, "Ni": 58.693,
               "V": 50.942}


def mol_from(smiles_or_name: str):
    from hydromcp.chemistry.library import MODEL_COMPOUNDS

    key = smiles_or_name.strip().lower()
    smiles = MODEL_COMPOUNDS[key]["smiles"] if key in MODEL_COMPOUNDS else smiles_or_name
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse '{smiles_or_name}' as SMILES or a known model compound")
    return mol


def element_counts(mol) -> Counter:
    molh = Chem.AddHs(mol)
    return Counter(a.GetSymbol() for a in molh.GetAtoms())


def molecule_properties(smiles_or_name: str) -> dict:
    mol = mol_from(smiles_or_name)
    counts = element_counts(mol)
    mw = Descriptors.MolWt(mol)
    wt = {el: n * ATOMIC_MASS.get(el, 0) / mw * 100 for el, n in counts.items()}
    c, h = counts.get("C", 0), counts.get("H", 0)
    n_, hal = counts.get("N", 0), 0
    dbe = c - h / 2 + n_ / 2 + 1 - hal / 2
    ri = mol.GetRingInfo()
    arom_rings = sum(1 for ring in ri.AtomRings()
                     if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    return {
        "input": smiles_or_name,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": mw,
        "exact_mass": Descriptors.ExactMolWt(mol),
        "element_counts": dict(counts),
        "element_wt_pct": {k: round(v, 4) for k, v in wt.items()},
        "h_to_c_ratio": h / c if c else None,
        "double_bond_equivalents": dbe,
        "ring_count": ri.NumRings(),
        "aromatic_ring_count": arom_rings,
    }


def validate_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    return {"valid": mol is not None,
            "canonical_smiles": Chem.MolToSmiles(mol) if mol else None}


def similarity(a: str, b: str) -> float:
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return DataStructs.TanimotoSimilarity(gen.GetFingerprint(mol_from(a)),
                                          gen.GetFingerprint(mol_from(b)))
