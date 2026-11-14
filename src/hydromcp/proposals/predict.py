"""Calibration on the existing unit, performance/yield prediction for proposals, comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import brentq

from hydromcp.core.library import Library, get_library
from hydromcp.core.schemas import OperatingBasis, ProcessUnit, Proposal
from hydromcp.core.units import LB_PER_BBL_WATER, SCF_PER_LBMOL
from hydromcp.loading.loading import plan_loading
from hydromcp.proposals.kinetics import T_REF_F, LayerModel, SystemModel

# Stoichiometric H2 defaults (mol H2 per mol heteroatom / aromatic). The chemistry tools
# (h2_demand) compute these for specific model compounds if you want to override.
DEFAULT_H2_STOICH = {
    "h2_per_mol_s": 3.0,       # DBT-type, mix of direct desulfurization and hydrogenation
    "h2_per_mol_n": 5.0,       # quinoline/carbazole type HDN
    "h2_per_mol_aromatic": 3.0,  # mono-aromatic ring saturation
    "aromatic_mw": 160.0,
    "h2_scf_bbl_per_wt_conversion": 18.0,  # hydrocracking, beyond hydrotreating demand
}


def build_system(proposal: Proposal, unit: ProcessUnit, lib: Library) -> tuple[SystemModel, dict]:
    plan = plan_loading(proposal, unit, lib)
    layers = [LayerModel(key=f"{lr.reactor_id}/{lr.bed_name}", catalyst=lib.catalyst(lr.catalyst_id),
                         volume_ft3=lr.volume_ft3) for lr in plan["_layers"]]
    return SystemModel(layers=layers), plan


@dataclass
class Calibration:
    unit_id: str
    proposal_id: str
    k_ref: dict[str, float]
    t_ref_f: float
    h2pp_ref_psia: float
    age_months: float
    basis: dict
    fit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(unit: ProcessUnit, loading: Proposal, basis: OperatingBasis,
              age_months: float = 0.0, lib: Library | None = None) -> Calibration:
    """Fit k_ref per reaction so the model reproduces the unit's measured performance.

    Uses basis.wabt_f with measured product_sulfur_wppm (HDS), product_nitrogen_wppm (HDN),
    product_aromatics_wt (HDA) and conversion_wt (HCK) where given. age_months is the
    catalyst age at the time of the data, so k_ref represents fresh catalyst."""
    lib = lib or get_library()
    if basis.wabt_f is None:
        raise ValueError("Calibration needs the actual WABT (basis.wabt_f)")
    system, _ = build_system(loading, unit, lib)
    system.h2pp_ref_psia = basis.h2_partial_pressure_psia
    targets = {}
    if basis.product_sulfur_wppm:
        targets["hds"] = (basis.feed_sulfur_wppm, basis.product_sulfur_wppm)
    if basis.product_nitrogen_wppm and basis.feed_nitrogen_wppm:
        targets["hdn"] = (basis.feed_nitrogen_wppm, basis.product_nitrogen_wppm)
    if basis.product_aromatics_wt is not None and basis.feed_aromatics_wt:
        targets["hda"] = (basis.feed_aromatics_wt, basis.product_aromatics_wt)
    if basis.conversion_wt is not None and basis.feed_conversion_fraction_wt:
        heavy_in = basis.feed_conversion_fraction_wt
        targets["hck"] = (heavy_in, heavy_in * (1 - basis.conversion_wt / 100.0))
    if not targets:
        raise ValueError("Give at least one measured product property (e.g. product_sulfur_wppm)")
    k_ref, fit = {}, {}
    for rxn, (c_in, c_out) in targets.items():
        if not any(lm.catalyst.rva(rxn) > 0 for lm in system.layers):
            fit[rxn] = "skipped: no catalyst in the loading has activity for this reaction"
            continue
        g = lambda lk: np.log(system.outlet(rxn, c_in, basis.wabt_f, basis, age_months,
                                            k_ref=np.exp(lk))) - np.log(c_out)
        lk = brentq(g, -30.0, 30.0, xtol=1e-8)
        k_ref[rxn] = float(np.exp(lk))
        fit[rxn] = {"c_in": c_in, "c_out_measured": c_out}
    return Calibration(unit_id=unit.id, proposal_id=loading.id, k_ref=k_ref, t_ref_f=T_REF_F,
                       h2pp_ref_psia=basis.h2_partial_pressure_psia, age_months=age_months,
                       basis=basis.model_dump(exclude_none=True), fit=fit)


def _h2_and_yields(system: SystemModel, basis: OperatingBasis, s_out: float, n_out: float | None,
                   arom_out: float | None, heavy_out: float | None, stoich: dict) -> dict:
    feed_lb_bbl = basis.feed_sg * LB_PER_BBL_WATER
    s_rem_wt = (basis.feed_sulfur_wppm - s_out) * 1e-4  # wt% of feed
    n_rem_wt = ((basis.feed_nitrogen_wppm - n_out) * 1e-4) if n_out is not None else 0.0
    arom_sat_wt = (basis.feed_aromatics_wt - arom_out) if arom_out is not None else 0.0
    converted_wt = ((basis.feed_conversion_fraction_wt - heavy_out)
                    if heavy_out is not None else 0.0)

    h2 = {
        "hds": s_rem_wt / 100 * feed_lb_bbl / 32.06 * stoich["h2_per_mol_s"] * SCF_PER_LBMOL,
        "hdn": n_rem_wt / 100 * feed_lb_bbl / 14.007 * stoich["h2_per_mol_n"] * SCF_PER_LBMOL,
        "hda": arom_sat_wt / 100 * feed_lb_bbl / stoich["aromatic_mw"]
               * stoich["h2_per_mol_aromatic"] * SCF_PER_LBMOL,
        "hck": converted_wt * stoich["h2_scf_bbl_per_wt_conversion"],
    }
    vols = [lm.volume_ft3 for lm in system.active_layers]
    factor = (sum(lm.catalyst.h2_consumption_factor * lm.volume_ft3 for lm in system.active_layers)
              / sum(vols)) if vols else 1.0
    h2_total = sum(h2.values()) * factor
    h2_wt = h2_total / SCF_PER_LBMOL * 2.016 / feed_lb_bbl * 100.0

    yields = {"H2S": s_rem_wt * 34.08 / 32.06, "NH3": n_rem_wt * 17.03 / 14.007}
    sel_layers = [lm for lm in system.layers if lm.catalyst.yield_selectivity and lm.catalyst.rva("hck") > 0]
    if converted_wt > 0:
        if sel_layers:
            w = {}
            tot = sum(lm.catalyst.rva("hck") * lm.volume_ft3 for lm in sel_layers)
            for lm in sel_layers:
                for cut, frac in lm.catalyst.yield_selectivity.items():
                    w[cut] = w.get(cut, 0.0) + frac * lm.catalyst.rva("hck") * lm.volume_ft3 / tot
            for cut, frac in w.items():
                yields[cut] = yields.get(cut, 0.0) + converted_wt * frac
        else:
            yields["converted_unallocated"] = converted_wt
    yields["hydrotreated_liquid" if converted_wt == 0 else "unconverted_and_treated_liquid"] = (
        100.0 - s_rem_wt - n_rem_wt - converted_wt
        + (h2_wt - yields["H2S"] + s_rem_wt - yields["NH3"] + n_rem_wt))
    return {
        "h2_consumption_scf_bbl": h2_total,
        "h2_consumption_by_reaction_scf_bbl": {k: v * factor for k, v in h2.items() if v},
        "h2_consumption_wt_pct": h2_wt,
        "yields_wt_pct_of_feed": yields,
        "yield_total_wt_pct": sum(yields.values()),
        "conversion_wt_pct": (converted_wt / basis.feed_conversion_fraction_wt * 100.0
                              if basis.feed_conversion_fraction_wt else None),
    }


def predict(proposal: Proposal, calibration: Calibration, basis: OperatingBasis,
            unit: ProcessUnit | None = None, lib: Library | None = None,
            product_sulfur_target_wppm: float | None = None, age_months: float = 0.0,
            max_wabt_f: float | None = None, simulate_cycle: bool = True,
            stoich: dict | None = None, max_months: int = 120) -> dict:
    """Predict performance of a catalyst system at given operating conditions.

    If product_sulfur_target_wppm (or unit spec) is given, solves the WABT needed at
    age_months and simulates the cycle until max WABT. Otherwise uses basis.wabt_f."""
    lib = lib or get_library()
    unit = unit or lib.unit(proposal.unit_id)
    stoich = {**DEFAULT_H2_STOICH, **(stoich or {})}
    system, plan = build_system(proposal, unit, lib)
    system.k_ref = dict(calibration.k_ref)
    system.t_ref_f = calibration.t_ref_f
    system.h2pp_ref_psia = calibration.h2pp_ref_psia
    max_wabt = max_wabt_f or unit.max_wabt_f
    target = product_sulfur_target_wppm or unit.product_sulfur_spec_wppm
    warnings = list(plan["warnings"])

    if "hds" not in system.k_ref:
        raise ValueError("Calibration has no HDS rate constant")
    if target:
        wabt = system.required_wabt("hds", basis.feed_sulfur_wppm, target, basis, age_months)
    elif basis.wabt_f is not None:
        wabt = basis.wabt_f
    else:
        raise ValueError("Give product_sulfur_target_wppm (or a unit spec) or basis.wabt_f")

    s_out = system.outlet("hds", basis.feed_sulfur_wppm, wabt, basis, age_months)
    n_out = (system.outlet("hdn", basis.feed_nitrogen_wppm, wabt, basis, age_months)
             if "hdn" in system.k_ref and basis.feed_nitrogen_wppm else None)
    arom_out = (system.outlet("hda", basis.feed_aromatics_wt, wabt, basis, age_months)
                if "hda" in system.k_ref and basis.feed_aromatics_wt else None)
    heavy_out = (system.outlet("hck", basis.feed_conversion_fraction_wt, wabt, basis, age_months)
                 if "hck" in system.k_ref and basis.feed_conversion_fraction_wt else None)

    for lm in system.layers:
        tmax = lm.catalyst.max_temperature_f
        if tmax and wabt > tmax:
            warnings.append(f"WABT {wabt:.0f} F exceeds max temperature {tmax:.0f} F for "
                            f"{lm.catalyst.id}")
    if wabt > max_wabt:
        warnings.append(f"Required WABT {wabt:.1f} F exceeds unit max {max_wabt:.0f} F")

    result = {
        "proposal_id": proposal.id,
        "calibration_unit": calibration.unit_id,
        "age_months": age_months,
        "wabt_f": wabt,
        "product_sulfur_wppm": s_out,
        "product_nitrogen_wppm": n_out,
        "product_aromatics_wt": arom_out,
        "lhsv_active_1_hr": basis.feed_rate_bpd * 5.614583 / 24.0 / plan["totals"]["active_volume_ft3"]
        if plan["totals"]["active_volume_ft3"] else None,
        "active_volume_ft3": plan["totals"]["active_volume_ft3"],
        "active_mass_lb": plan["totals"]["active_mass_lb"],
        "composite_deactivation_f_per_month": system.weighted(
            lambda c: c.deactivation_rate_f_per_month, "hds"),
        **_h2_and_yields(system, basis, s_out, n_out, arom_out, heavy_out, stoich),
    }
    cost = sum(o.get("estimated_cost", 0.0) for o in plan["order_summary"])
    if cost:
        result["estimated_catalyst_cost"] = cost

    if simulate_cycle and target:
        trajectory, cycle = [], None
        prev = None
        for m in range(max_months + 1):
            try:
                t = system.required_wabt("hds", basis.feed_sulfur_wppm, target, basis, float(m))
            except ValueError:
                t = float("inf")
            if m % 3 == 0 and np.isfinite(t):
                trajectory.append({"month": m, "wabt_f": round(t, 2)})
            if t >= max_wabt:
                if prev is None or not np.isfinite(t):
                    cycle = float(m)
                else:
                    cycle = (m - 1) + (max_wabt - prev) / (t - prev)
                break
            prev = t
        result["cycle_length_months"] = cycle if cycle is not None else f">{max_months}"
        result["max_wabt_f"] = max_wabt
        result["wabt_trajectory"] = trajectory
    result["warnings"] = warnings
    return result


def compare(results: list[dict], base_index: int = 0) -> dict:
    """Tabulate predictions; deltas relative to results[base_index]."""
    base = results[base_index]
    keys = ["wabt_f", "product_sulfur_wppm", "product_nitrogen_wppm", "cycle_length_months",
            "h2_consumption_scf_bbl", "active_volume_ft3", "active_mass_lb",
            "conversion_wt_pct", "estimated_catalyst_cost"]
    rows = []
    for r in results:
        row = {"proposal_id": r["proposal_id"]}
        for k in keys:
            if r.get(k) is not None:
                row[k] = r[k]
                b = base.get(k)
                if isinstance(r[k], (int, float)) and isinstance(b, (int, float)):
                    row[f"delta_{k}"] = r[k] - b
        row["yields_wt_pct_of_feed"] = r.get("yields_wt_pct_of_feed")
        rows.append(row)

    def cyc(r):
        c = r.get("cycle_length_months")
        return c if isinstance(c, (int, float)) else 1e9

    ranking = sorted(results, key=lambda r: (-cyc(r), r["wabt_f"]))
    return {"base": base["proposal_id"], "table": rows,
            "ranking_by_cycle_then_wabt": [r["proposal_id"] for r in ranking]}
