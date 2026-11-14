"""Reaction atom balances and hydrogen demand using RDKit formulas."""

from __future__ import annotations

from collections import Counter

from hydromcp.chemistry.library import PATHWAYS
from hydromcp.chemistry.molecules import element_counts, mol_from, molecule_properties
from hydromcp.core.units import SCF_PER_LBMOL


def _counts(species: list[tuple[str, float]]) -> Counter:
    total: Counter = Counter()
    for s, coeff in species:
        for el, n in element_counts(mol_from(s)).items():
            total[el] += n * coeff
    return total


def check_balance(reactants: list[dict], products: list[dict]) -> dict:
    """reactants/products: [{"species": smiles or name, "coeff": 1}, ...]."""
    r = _counts([(x["species"], x.get("coeff", 1)) for x in reactants])
    p = _counts([(x["species"], x.get("coeff", 1)) for x in products])
    els = sorted(set(r) | set(p))
    diff = {el: p.get(el, 0) - r.get(el, 0) for el in els}
    return {"balanced": all(abs(v) < 1e-9 for v in diff.values()),
            "reactant_atoms": dict(r), "product_atoms": dict(p),
            "imbalance_products_minus_reactants": {k: v for k, v in diff.items() if v}}


def h2_demand(reactant: str | None = None, products: list[str] | None = None,
              pathway: str | None = None) -> dict:
    """Moles of H2 per mole of reactant to reach the given products (1 mol each).

    Either a named pathway (see PATHWAYS) or reactant + products. Non-H elements must
    balance; H2 is solved from the H imbalance."""
    if pathway:
        if pathway not in PATHWAYS:
            raise KeyError(f"Unknown pathway '{pathway}'. Known: {sorted(PATHWAYS)}")
        reactant, products = PATHWAYS[pathway]["reactant"], PATHWAYS[pathway]["products"]
    if not reactant or not products:
        raise ValueError("Give a pathway or reactant + products")
    r = _counts([(reactant, 1)])
    p = _counts([(s, 1) for s in products])
    other = {el: p.get(el, 0) - r.get(el, 0) for el in set(r) | set(p) if el != "H"}
    if any(abs(v) > 1e-9 for v in other.values()):
        raise ValueError(f"Non-hydrogen atoms do not balance: {other}")
    n_h2 = (p.get("H", 0) - r.get("H", 0)) / 2.0
    props = molecule_properties(reactant)
    mw = props["molecular_weight"]
    hetero = {el: props["element_counts"].get(el, 0) for el in ("S", "N", "O")}
    out = {
        "reactant": props["formula"],
        "products": [molecule_properties(s)["formula"] for s in products],
        "h2_mol_per_mol": n_h2,
        "h2_scf_per_lb_reactant": n_h2 * SCF_PER_LBMOL / mw,
        "equation": f"{props['formula']} + {n_h2:g} H2 -> "
                    + " + ".join(molecule_properties(s)["formula"] for s in products),
    }
    for el, n in hetero.items():
        if n:
            out[f"h2_mol_per_mol_{el}"] = n_h2 / n
    if pathway and PATHWAYS[pathway].get("notes"):
        out["notes"] = PATHWAYS[pathway]["notes"]
    return out


def h2_for_feed(feed_sg: float, s_removed_wppm: float = 0.0, n_removed_wppm: float = 0.0,
                s_pathway: str = "dbt_dds", n_pathway: str = "quinoline_hdn") -> dict:
    """Chemical H2 (SCF/bbl) for S and N removal using model-compound stoichiometry."""
    from hydromcp.core.units import LB_PER_BBL_WATER

    lb_bbl = feed_sg * LB_PER_BBL_WATER
    s = h2_demand(pathway=s_pathway).get("h2_mol_per_mol_S", 0.0)
    n = h2_demand(pathway=n_pathway).get("h2_mol_per_mol_N", 0.0)
    h2_s = lb_bbl * s_removed_wppm * 1e-6 / 32.06 * s * SCF_PER_LBMOL
    h2_n = lb_bbl * n_removed_wppm * 1e-6 / 14.007 * n * SCF_PER_LBMOL
    return {"h2_per_mol_s": s, "h2_per_mol_n": n, "hds_scf_bbl": h2_s, "hdn_scf_bbl": h2_n,
            "total_scf_bbl": h2_s + h2_n}
