"""Layer-by-layer plug-flow kinetic model of a catalyst system.

For each reaction r (hds, hdn, hda, hck) and each layer j (flow order):

    k_rj = k_ref,r * RVA_rj * exp[-Ea_rj/R (1/T_j - 1/T_ref)] * (PH2/PH2_ref)^alpha_rj
    nth order (n != 1):  C_out^(1-n) = C_in^(1-n) + (n-1) k / LHSV_j
    first order:         C_out = C_in exp(-k / LHSV_j)

LHSV_j = feed volumetric rate / layer volume, so layers act in series.

Deactivation: each catalyst's deactivation_rate_f_per_month is applied as an effective
temperature loss, T_eff,j = T_j - rate_j * age_months (the catalyst at age t behaves as
fresh catalyst at a lower temperature). Different catalysts therefore age at their own rates.

k_ref is calibrated on the existing unit (calibrate.py), so RVA values must be expressed
relative to the same reference catalyst (reference RVA = 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from hydromcp.core.schemas import REACTIONS, Catalyst, OperatingBasis
from hydromcp.core.units import FT3_PER_BBL, R_KJ, f_to_k

T_REF_F = 700.0


@dataclass
class LayerModel:
    key: str  # "reactor_id/bed_name"
    catalyst: Catalyst
    volume_ft3: float


@dataclass
class SystemModel:
    layers: list[LayerModel]
    k_ref: dict[str, float] = field(default_factory=dict)  # fresh-catalyst basis
    t_ref_f: float = T_REF_F
    h2pp_ref_psia: float = 1000.0

    @property
    def active_layers(self) -> list[LayerModel]:
        return [lm for lm in self.layers if any(lm.catalyst.rva(r) > 0 for r in REACTIONS)]

    def layer_k(self, lm: LayerModel, rxn: str, t_f: float, h2pp: float, k_ref: float) -> float:
        cat = lm.catalyst
        rva = cat.rva(rxn)
        if rva <= 0:
            return 0.0
        arr = np.exp(-cat.ea(rxn) / R_KJ * (1.0 / f_to_k(t_f) - 1.0 / f_to_k(self.t_ref_f)))
        return k_ref * rva * arr * (h2pp / self.h2pp_ref_psia) ** cat.h2pp_exp(rxn)

    def layer_temps(self, wabt_f: float, basis: OperatingBasis, age_months: float) -> list[float]:
        offsets = basis.bed_temperature_offsets_f or {}
        return [wabt_f + offsets.get(lm.key, 0.0)
                - lm.catalyst.deactivation_rate_f_per_month * age_months for lm in self.layers]

    def outlet(self, rxn: str, c_in: float, wabt_f: float, basis: OperatingBasis,
               age_months: float = 0.0, k_ref: float | None = None) -> float:
        """Outlet concentration of species for reaction rxn after all layers."""
        kr = self.k_ref.get(rxn) if k_ref is None else k_ref
        if kr is None:
            raise ValueError(f"No calibrated k_ref for '{rxn}'")
        q_ft3_hr = basis.feed_rate_bpd * FT3_PER_BBL / 24.0
        c = float(c_in)
        temps = self.layer_temps(wabt_f, basis, age_months)
        for lm, t in zip(self.layers, temps):
            k = self.layer_k(lm, rxn, t, basis.h2_partial_pressure_psia, kr)
            if k <= 0 or lm.volume_ft3 <= 0:
                continue
            lhsv = q_ft3_hr / lm.volume_ft3
            n = lm.catalyst.order(rxn)
            if abs(n - 1.0) < 1e-9:
                c = c * np.exp(-k / lhsv)
            else:
                c = (c ** (1.0 - n) + (n - 1.0) * k / lhsv) ** (1.0 / (1.0 - n))
            if c <= 1e-12:
                return 1e-12
        return c

    def required_wabt(self, rxn: str, c_in: float, c_target: float, basis: OperatingBasis,
                      age_months: float = 0.0, t_lo: float = 450.0, t_hi: float = 950.0) -> float:
        """WABT that gives outlet c_target (raises if outside the bracket)."""
        f = lambda t: np.log(self.outlet(rxn, c_in, t, basis, age_months)) - np.log(c_target)
        if f(t_lo) < 0:
            return t_lo
        if f(t_hi) > 0:
            raise ValueError(f"Target {c_target} for {rxn} not reachable below {t_hi} F")
        return brentq(f, t_lo, t_hi, xtol=0.01)

    def weighted(self, attr_fn, rxn: str = "hds", default: float = 0.0) -> float:
        """Activity-volume weighted average of a per-catalyst attribute over active layers."""
        w = [(lm.catalyst.rva(rxn) * lm.volume_ft3, attr_fn(lm.catalyst)) for lm in self.layers]
        total = sum(x for x, _ in w)
        return sum(x * v for x, v in w) / total if total > 0 else default
