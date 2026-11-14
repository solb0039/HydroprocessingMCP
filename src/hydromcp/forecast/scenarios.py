"""What-if scenarios on a calibrated unit: change feed rate, feed quality, H2PP, product
target or catalyst age and see the required WABT, product quality and remaining cycle."""

from __future__ import annotations

from hydromcp.core.library import Library, get_library
from hydromcp.core.schemas import OperatingBasis
from hydromcp.proposals.predict import Calibration, predict


def what_if(calibration: Calibration, changes: dict, age_months: float | None = None,
            product_sulfur_target_wppm: float | None = None, proposal_id: str | None = None,
            lib: Library | None = None) -> dict:
    lib = lib or get_library()
    unit = lib.unit(calibration.unit_id)
    proposal = lib.proposal(proposal_id or calibration.proposal_id)
    base = OperatingBasis.model_validate(calibration.basis)
    new = OperatingBasis.model_validate({**calibration.basis, **changes})
    age = calibration.age_months if age_months is None else age_months
    target = product_sulfur_target_wppm or base.product_sulfur_wppm
    r_base = predict(proposal, calibration, base, unit, lib, target, age)
    r_new = predict(proposal, calibration, new, unit, lib, target, age)

    def remaining(r):
        c = r.get("cycle_length_months")
        return c - age if isinstance(c, (int, float)) else c

    return {
        "changes": changes,
        "age_months": age,
        "product_sulfur_target_wppm": target,
        "base": {"wabt_f": r_base["wabt_f"], "cycle_length_months": r_base.get("cycle_length_months"),
                 "remaining_months": remaining(r_base),
                 "h2_consumption_scf_bbl": r_base["h2_consumption_scf_bbl"]},
        "scenario": {"wabt_f": r_new["wabt_f"], "cycle_length_months": r_new.get("cycle_length_months"),
                     "remaining_months": remaining(r_new),
                     "h2_consumption_scf_bbl": r_new["h2_consumption_scf_bbl"],
                     "product_nitrogen_wppm": r_new.get("product_nitrogen_wppm")},
        "delta_wabt_f": r_new["wabt_f"] - r_base["wabt_f"],
        "warnings": r_new["warnings"],
    }
