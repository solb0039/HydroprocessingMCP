"""RDKit-based chemistry tools for hydroprocessing model compounds."""

from __future__ import annotations

from fastmcp import FastMCP

from hydromcp.chemistry import library, molecules, stoichiometry

mcp = FastMCP("chemistry")


@mcp.tool
def molecule_properties(smiles_or_name: str) -> dict:
    """Formula, MW, element counts and wt% (S, N, H...), H/C, double-bond equivalents and
    aromatic rings for a SMILES string or a model compound name (e.g. 'dibenzothiophene')."""
    return molecules.molecule_properties(smiles_or_name)


@mcp.tool
def validate_smiles(smiles: str) -> dict:
    """Check a SMILES string and return its canonical form."""
    return molecules.validate_smiles(smiles)


@mcp.tool
def model_compounds(query: str | None = None) -> list[dict]:
    """List hydroprocessing model compounds, filtered by name or class (sulfur, nitrogen,
    aromatic, product, gas, paraffin)."""
    return library.lookup(query)


@mcp.tool
def list_reaction_pathways() -> dict:
    """Named HDS/HDN/aromatic saturation pathways usable with h2_demand."""
    return library.PATHWAYS


@mcp.tool
def h2_demand(pathway: str | None = None, reactant: str | None = None,
              products: list[str] | None = None) -> dict:
    """Hydrogen required (mol H2/mol and SCF/lb) for a reaction, from an RDKit atom balance.
    Use a named pathway (e.g. 'dbt_hyd') or a reactant plus product list."""
    return stoichiometry.h2_demand(reactant, products, pathway)


@mcp.tool
def check_reaction_balance(reactants: list[dict], products: list[dict]) -> dict:
    """Atom balance check. reactants/products: [{'species': SMILES or name, 'coeff': 1}]."""
    return stoichiometry.check_balance(reactants, products)


@mcp.tool
def h2_for_feed(feed_sg: float, s_removed_wppm: float = 0.0, n_removed_wppm: float = 0.0,
                s_pathway: str = "dbt_dds", n_pathway: str = "quinoline_hdn") -> dict:
    """Chemical H2 (SCF/bbl) for sulfur and nitrogen removal using model-compound
    stoichiometry; the mol-H2 ratios can be fed into predict_performance h2_stoichiometry."""
    return stoichiometry.h2_for_feed(feed_sg, s_removed_wppm, n_removed_wppm, s_pathway, n_pathway)


@mcp.tool
def molecular_similarity(a: str, b: str) -> dict:
    """Tanimoto similarity (Morgan r=2) between two molecules."""
    return {"tanimoto": molecules.similarity(a, b)}
