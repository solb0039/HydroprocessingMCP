"""Packed-bed pressure drop (Ergun). For trickle-bed reactors this is a clean-bed screening
index (sum of single-phase gas and liquid Ergun terms), useful for comparing loadings and
grading schemes; it is not a rigorous two-phase design method."""

from __future__ import annotations

import math

LBHR_TO_KGS = 0.45359237 / 3600.0
LBFT3_TO_KGM3 = 16.018463
FT_TO_M = 0.3048
PA_TO_PSI = 1.0 / 6894.757

# equivalent-sphere diameter (6 Vp / Sp) relative to nominal diameter, typical L/D ~ 3
SHAPE_FACTORS = {"sphere": 1.0, "cylinder": 1.29, "extrudate": 1.29, "trilobe": 1.10,
                 "quadlobe": 1.05, "ring": 0.80, "rings": 0.80}


def equivalent_diameter_mm(size_mm: float, shape: str | None) -> float:
    return size_mm * SHAPE_FACTORS.get((shape or "cylinder").lower(), 1.0)


def ergun_dp_per_length(mass_flow_lb_hr: float, density_lb_ft3: float, viscosity_cp: float,
                        area_ft2: float, dp_mm: float, void_fraction: float) -> float:
    """Ergun pressure gradient in psi/ft for one phase flowing alone through the bed."""
    rho = density_lb_ft3 * LBFT3_TO_KGM3
    mu = viscosity_cp * 1e-3
    u = mass_flow_lb_hr * LBHR_TO_KGS / rho / (area_ft2 * FT_TO_M ** 2)
    dp = dp_mm / 1000.0
    e = void_fraction
    grad = (150.0 * mu * (1 - e) ** 2 * u / (e ** 3 * dp ** 2)
            + 1.75 * rho * (1 - e) * u ** 2 / (e ** 3 * dp))  # Pa/m
    return grad * PA_TO_PSI * FT_TO_M


def layer_pressure_drop(depth_ft: float, area_ft2: float, dp_mm: float, void_fraction: float,
                        liquid_lb_hr: float = 0.0, liquid_density_lb_ft3: float = 50.0,
                        liquid_viscosity_cp: float = 0.3, gas_lb_hr: float = 0.0,
                        gas_density_lb_ft3: float = 1.0, gas_viscosity_cp: float = 0.015) -> dict:
    liq = (ergun_dp_per_length(liquid_lb_hr, liquid_density_lb_ft3, liquid_viscosity_cp,
                               area_ft2, dp_mm, void_fraction) * depth_ft if liquid_lb_hr else 0.0)
    gas = (ergun_dp_per_length(gas_lb_hr, gas_density_lb_ft3, gas_viscosity_cp, area_ft2,
                               dp_mm, void_fraction) * depth_ft if gas_lb_hr else 0.0)
    return {"dp_liquid_psi": liq, "dp_gas_psi": gas, "dp_total_psi": liq + gas}


def bed_area_ft2(diameter_ft: float) -> float:
    return math.pi * diameter_ft ** 2 / 4.0
