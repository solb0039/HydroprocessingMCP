"""Unit handling. Internally the library works in US refinery field units:

temperature degF, pressure psia (gauge inputs converted by +14.696), feed rate BPD,
gas rates SCF (60 degF, 14.696 psia), density lb/ft3, length ft, catalyst mass lb.
"""

from __future__ import annotations

import math

import pint

ureg = pint.UnitRegistry()
ureg.define("BPD = barrel / day")
ureg.define("SCF = foot ** 3")  # standard cubic foot, volume at 60 degF / 1 atm
ureg.define("MSCF = 1000 * SCF")
ureg.define("MMSCF = 1e6 * SCF")
Q_ = ureg.Quantity

ATM_PSI = 14.696
FT3_PER_BBL = 5.614583
LB_PER_BBL_WATER = 350.17  # lb of 60 degF water per barrel; multiply by SG for oil
SCF_PER_LBMOL = 379.49  # ideal gas at 60 degF, 14.696 psia
R_KJ = 8.314462618e-3  # kJ/(mol K)


def f_to_k(t_f):
    return (t_f - 32.0) * 5.0 / 9.0 + 273.15


def k_to_f(t_k):
    return (t_k - 273.15) * 9.0 / 5.0 + 32.0


def f_to_c(t_f):
    return (t_f - 32.0) * 5.0 / 9.0


def c_to_f(t_c):
    return t_c * 9.0 / 5.0 + 32.0


def psig_to_psia(p):
    return p + ATM_PSI


def sg_to_api(sg):
    return 141.5 / sg - 131.5


def api_to_sg(api):
    return 141.5 / (api + 131.5)


_GAUGE = {"psig": ("psi", ATM_PSI), "barg": ("bar", 1.01325), "kPag": ("kPa", 101.325)}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units. Handles gauge pressures (psig, barg, kPag)."""
    offset_in = 0.0
    if from_unit in _GAUGE:
        from_unit, offset_in = _GAUGE[from_unit]
    to_offset = 0.0
    if to_unit in _GAUGE:
        to_unit, to_offset = _GAUGE[to_unit]
    q = Q_(value + offset_in, from_unit).to(to_unit).magnitude
    return float(q - to_offset)


def is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
