"""Section 4 MCP tools: catalyst library, catalyst-system proposals, calibration and
performance / yield predictions."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from hydromcp.core.equations import to_jsonable
from hydromcp.core.library import get_library
from hydromcp.core.schemas import Catalyst, OperatingBasis, Proposal
from hydromcp.core.state import STORE
from hydromcp.loading.loading import plan_loading, public
from hydromcp.proposals import predict as pred

mcp = FastMCP("proposals")


@mcp.tool
def list_catalysts(function: str | None = None) -> list[dict]:
    """List catalysts in the library (optionally by function: grading, hdm, hds, hdn,
    pretreat, hck, ...) with their key performance values."""
    return [{"id": c.id, "name": c.name, "vendor": c.vendor, "function": c.function,
             "metals": c.metals, "shape": c.shape, "size_mm": c.size_mm,
             "density_sock_lb_ft3": c.density_sock_lb_ft3, "activity": c.activity,
             "deactivation_rate_f_per_month": c.deactivation_rate_f_per_month}
            for c in get_library().catalysts() if function is None or c.function == function]


@mcp.tool
def get_catalyst(catalyst_id: str) -> dict:
    """Full performance definition of a catalyst."""
    return get_library().catalyst(catalyst_id).model_dump(exclude_none=True)


@mcp.tool
def add_catalyst(catalyst: Catalyst, persist: bool = True) -> dict:
    """Add or replace a catalyst in the library (saved to data/catalysts/user_catalysts.yaml).
    activity values are relative volume activities vs the reference catalyst (RVA = 1.0)."""
    get_library().add_catalyst(catalyst, persist)
    return {"added": catalyst.id, "persisted": persist}


@mcp.tool
def reload_library() -> dict:
    """Re-read catalysts, units and proposals from the data folder."""
    lib = get_library()
    lib.reload()
    return {"catalysts": len(lib.catalysts()), "units": len(lib.units()),
            "proposals": len(lib.proposals()), "errors": lib.errors}


@mcp.tool
def create_proposal(proposal: Proposal, persist: bool = True, contingency_pct: float = 5.0) -> dict:
    """Define a catalyst system for an existing unit: for each bed (in flow order) the
    layers top-to-bottom with catalyst_id, loading_method and one of depth_ft, volume_ft3,
    mass_lb or bed_fraction. Validates the fit and returns the loading plan."""
    lib = get_library()
    lib.unit(proposal.unit_id)
    plan = plan_loading(proposal, contingency_pct=contingency_pct)
    lib.add_proposal(proposal, persist)
    return to_jsonable({"proposal_id": proposal.id, "persisted": persist,
                        "loading": public(plan)})


@mcp.tool
def list_proposals(unit_id: str | None = None) -> list[dict]:
    """List saved proposals (use '<unit_id>:current' to refer to a unit's existing loading)."""
    return [{"id": p.id, "name": p.name, "unit_id": p.unit_id,
             "beds": [f"{b.reactor_id}/{b.bed_name}: " + ", ".join(l.catalyst_id for l in b.layers)
                      for b in p.beds]} for p in get_library().proposals(unit_id)]


@mcp.tool
def get_proposal(proposal_id: str) -> dict:
    """Full definition of a proposal (or '<unit_id>:current')."""
    return get_library().proposal(proposal_id).model_dump(exclude_none=True)


@mcp.tool
def calibrate_unit(unit_id: str, basis: OperatingBasis | None = None,
                   loading_id: str | None = None, age_months: float = 0.0) -> dict:
    """Calibrate the kinetic model to the unit's actual performance with its existing
    loading. basis must include wabt_f and measured product_sulfur_wppm (optionally
    nitrogen, aromatics, conversion). Defaults to the unit's design_basis. age_months is
    the catalyst age when the data was taken. Returns a calibration_id."""
    lib = get_library()
    unit = lib.unit(unit_id)
    basis = basis or unit.design_basis
    if basis is None:
        raise ValueError("No basis given and the unit has no design_basis")
    loading = lib.proposal(loading_id or f"{unit_id}:current")
    cal = pred.calibrate(unit, loading, basis, age_months, lib)
    cid = STORE.add_calibration(cal)
    return to_jsonable({"calibration_id": cid, **cal.to_dict()})


@mcp.tool
def calibrate_from_dataset(unit_id: str, dataset_id: str, column_map: dict[str, str],
                           start: str | None = None, end: str | None = None,
                           constants: dict[str, Any] | None = None,
                           age_column: str | None = "days_on_stream",
                           loading_id: str | None = None, statistic: str = "median") -> dict:
    """Calibrate using the median (or mean) of dataset columns over a window (e.g. a
    steady-state period). column_map maps OperatingBasis fields to columns, e.g.
    {'wabt_f': 'wabt', 'feed_rate_bpd': 'feed_rate', 'feed_sulfur_wppm': 'feed_s',
     'product_sulfur_wppm': 'prod_s', 'h2_partial_pressure_psia': 'h2pp'}.
    constants supplies fields not in the data (e.g. feed_sg). Catalyst age is taken from
    age_column (days) if present."""
    df = STORE.dataset(dataset_id).df.loc[start:end]
    agg = getattr(df, statistic)
    vals = {field: float(agg(numeric_only=True)[col]) for field, col in column_map.items()}
    basis = OperatingBasis.model_validate({**(constants or {}), **vals})
    age = 0.0
    if age_column and age_column in df.columns:
        age = float(agg(numeric_only=True)[age_column]) / 30.4375
    result = calibrate_unit(unit_id, basis, loading_id, age)
    result["window"] = {"start": str(df.index.min()), "end": str(df.index.max()), "rows": len(df)}
    return result


@mcp.tool
def predict_performance(proposal_id: str, calibration_id: str,
                        basis: OperatingBasis | None = None,
                        product_sulfur_target_wppm: float | None = None,
                        age_months: float = 0.0, max_wabt_f: float | None = None,
                        h2_stoichiometry: dict[str, float] | None = None) -> dict:
    """Predict a catalyst system's performance on the calibrated unit: SOR WABT needed for
    the product sulfur target, product S/N/aromatics, chemical H2 consumption, yields
    (wt% of feed incl. H2S/NH3 and hydrocracked cuts), composite deactivation rate and
    cycle length to max WABT. basis defaults to the calibration conditions."""
    lib = get_library()
    cal = STORE.calibration(calibration_id)
    basis = basis or OperatingBasis.model_validate(cal.basis)
    return to_jsonable(pred.predict(lib.proposal(proposal_id), cal, basis, None, lib,
                                    product_sulfur_target_wppm, age_months, max_wabt_f,
                                    stoich=h2_stoichiometry))


@mcp.tool
def compare_proposals(proposal_ids: list[str], calibration_id: str,
                      basis: OperatingBasis | None = None,
                      product_sulfur_target_wppm: float | None = None,
                      max_wabt_f: float | None = None) -> dict:
    """Predict and compare several catalyst systems (the first is the base case, e.g.
    '<unit_id>:current'). Ranks by cycle length then SOR WABT."""
    lib = get_library()
    cal = STORE.calibration(calibration_id)
    basis = basis or OperatingBasis.model_validate(cal.basis)
    target = product_sulfur_target_wppm or basis.product_sulfur_wppm
    results = [pred.predict(lib.proposal(pid), cal, basis, None, lib, target, 0.0, max_wabt_f)
               for pid in proposal_ids]
    out = pred.compare(results)
    out["warnings"] = {r["proposal_id"]: r["warnings"] for r in results if r["warnings"]}
    out["product_sulfur_target_wppm"] = target
    return to_jsonable(out)
