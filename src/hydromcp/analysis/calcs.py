"""Built-in hydroprocessing equations (Section 1). All are registered in the equation
registry and can be replaced by user equations of the same name.

Every function works element-wise on scalars, numpy arrays or pandas Series.
"""

from __future__ import annotations

import numpy as np

from hydromcp.core.equations import equation
from hydromcp.core.units import (
    ATM_PSI,
    FT3_PER_BBL,
    LB_PER_BBL_WATER,
    R_KJ,
    f_to_c,
    f_to_k,
    k_to_f,
)

# ---------------------------------------------------------------------------
# Temperatures
# ---------------------------------------------------------------------------


@equation(
    name="wabt_bed",
    category="temperature",
    inputs={"t_in": ("degF", "Bed inlet temperature"), "t_out": ("degF", "Bed outlet temperature")},
    outputs={"wabt": ("degF", "Bed weighted average bed temperature")},
    reference="Common industry approximation WABT = (Tin + 2*Tout)/3 for a linear-rise bed",
)
def wabt_bed(t_in, t_out):
    """Weighted average bed temperature of a single bed from inlet/outlet temperatures."""
    return (t_in + 2.0 * t_out) / 3.0


@equation(
    name="wabt_bed_multipoint",
    category="temperature",
    inputs={"temps": ("degF", "Bed temperatures top-to-bottom (inlet, internal TCs, outlet)")},
    outputs={"wabt": ("degF", "Trapezoidal average bed temperature")},
    list_inputs=("temps",),
    reference="Trapezoidal integration over equally spaced thermocouple elevations",
)
def wabt_bed_multipoint(temps):
    """Bed average temperature from several equally spaced thermocouple readings."""
    arr = [np.asarray(t, dtype=float) for t in temps]
    if len(arr) < 2:
        raise ValueError("Need at least two temperatures")
    total = sum(arr[1:-1]) + 0.5 * (arr[0] + arr[-1])
    return total / (len(arr) - 1)


@equation(
    name="wabt_reactor",
    category="temperature",
    inputs={
        "bed_wabts": ("degF", "WABT of each bed"),
        "bed_catalyst_weights": ("lb", "Catalyst weight (or volume) in each bed"),
    },
    outputs={"wabt": ("degF", "Catalyst-weighted reactor/unit WABT")},
    list_inputs=("bed_wabts", "bed_catalyst_weights"),
)
def wabt_reactor(bed_wabts, bed_catalyst_weights):
    """Catalyst weight weighted average of bed WABTs."""
    w = [np.asarray(x, dtype=float) for x in bed_catalyst_weights]
    t = [np.asarray(x, dtype=float) for x in bed_wabts]
    if len(w) != len(t):
        raise ValueError("bed_wabts and bed_catalyst_weights must have the same length")
    return sum(ti * wi for ti, wi in zip(t, w)) / sum(w)


@equation(
    name="bed_temperature_rise",
    category="temperature",
    inputs={"t_in": ("degF", "Bed inlet"), "t_out": ("degF", "Bed outlet")},
    outputs={"delta_t": ("degF", "Bed temperature rise")},
)
def bed_temperature_rise(t_in, t_out):
    """Bed temperature rise (exotherm)."""
    return t_out - t_in


# ---------------------------------------------------------------------------
# Space velocity and gas ratios
# ---------------------------------------------------------------------------


@equation(
    name="lhsv",
    category="space_velocity",
    inputs={
        "feed_rate_bpd": ("BPD", "Fresh feed rate"),
        "catalyst_volume_ft3": ("ft3", "Active catalyst volume"),
    },
    outputs={"lhsv": ("1/hr", "Liquid hourly space velocity")},
)
def lhsv(feed_rate_bpd, catalyst_volume_ft3):
    """LHSV = volumetric feed rate (ft3/h at 60F) / catalyst volume (ft3)."""
    return feed_rate_bpd * FT3_PER_BBL / 24.0 / catalyst_volume_ft3


@equation(
    name="whsv",
    category="space_velocity",
    inputs={
        "feed_rate_bpd": ("BPD", "Fresh feed rate"),
        "feed_sg": ("", "Feed specific gravity 60/60F"),
        "catalyst_mass_lb": ("lb", "Catalyst mass"),
    },
    outputs={"whsv": ("1/hr", "Weight hourly space velocity")},
)
def whsv(feed_rate_bpd, feed_sg, catalyst_mass_lb):
    """WHSV = mass feed rate (lb/h) / catalyst mass (lb)."""
    return feed_rate_bpd * feed_sg * LB_PER_BBL_WATER / 24.0 / catalyst_mass_lb


@equation(
    name="treat_gas_ratio",
    category="hydrogen",
    inputs={
        "treat_gas_mscfd": ("MSCF/day", "Treat gas rate to reactor"),
        "feed_rate_bpd": ("BPD", "Fresh feed rate"),
        "h2_purity_mol_frac": ("", "Treat gas H2 mole fraction", 1.0),
    },
    outputs={"h2_oil_ratio": ("SCF/bbl", "H2-to-oil ratio (pure H2 basis if purity given)")},
)
def treat_gas_ratio(treat_gas_mscfd, feed_rate_bpd, h2_purity_mol_frac=1.0):
    """Treat gas (or pure H2) to oil ratio in SCF/bbl."""
    return treat_gas_mscfd * 1000.0 * h2_purity_mol_frac / feed_rate_bpd


@equation(
    name="h2_partial_pressure",
    category="hydrogen",
    inputs={
        "pressure_psig": ("psig", "Reactor pressure (inlet or outlet)"),
        "h2_mol_frac": ("", "H2 mole fraction in gas phase"),
    },
    outputs={"h2_pp_psia": ("psia", "Hydrogen partial pressure")},
)
def h2_partial_pressure(pressure_psig, h2_mol_frac):
    """Hydrogen partial pressure = total pressure (psia) x H2 mole fraction."""
    return (pressure_psig + ATM_PSI) * h2_mol_frac


@equation(
    name="h2_consumption",
    category="hydrogen",
    inputs={
        "makeup_mscfd": ("MSCF/day", "Makeup gas rate"),
        "makeup_purity": ("", "Makeup H2 mole fraction"),
        "feed_rate_bpd": ("BPD", "Fresh feed rate"),
        "purge_mscfd": ("MSCF/day", "Purge/bleed gas rate", 0.0),
        "purge_purity": ("", "Purge H2 mole fraction", 0.0),
        "solution_loss_mscfd": ("MSCF/day", "H2 dissolved in separator liquids", 0.0),
    },
    outputs={"h2_consumption_scf_bbl": ("SCF/bbl", "Chemical H2 consumption")},
    reference="Hydrogen balance: makeup - purge - solution losses",
)
def h2_consumption(makeup_mscfd, makeup_purity, feed_rate_bpd, purge_mscfd=0.0,
                   purge_purity=0.0, solution_loss_mscfd=0.0):
    """Chemical hydrogen consumption from the unit H2 balance."""
    h2_in = makeup_mscfd * makeup_purity
    h2_out = purge_mscfd * purge_purity + solution_loss_mscfd
    return (h2_in - h2_out) * 1000.0 / feed_rate_bpd


# ---------------------------------------------------------------------------
# Conversion and kinetics
# ---------------------------------------------------------------------------


@equation(
    name="removal_pct",
    category="kinetics",
    inputs={"c_in": ("wppm", "Feed contaminant (S, N, ...)"),
            "c_out": ("wppm", "Product contaminant")},
    outputs={"removal_pct": ("%", "HDS/HDN removal")},
)
def removal_pct(c_in, c_out):
    """Percent removal (HDS, HDN, HDM ...)."""
    return (c_in - c_out) / c_in * 100.0


def _k_nth(c_in, c_out, lhsv_val, n):
    if abs(n - 1.0) < 1e-9:
        return lhsv_val * np.log(c_in / c_out)
    return lhsv_val / (n - 1.0) * (c_out ** (1.0 - n) - c_in ** (1.0 - n))


@equation(
    name="apparent_rate_constant",
    category="kinetics",
    inputs={
        "c_in": ("wppm", "Feed concentration"),
        "c_out": ("wppm", "Product concentration"),
        "lhsv": ("1/hr", "LHSV"),
        "order": ("", "Apparent reaction order (1.0 HDN, ~1.2-1.7 HDS)", 1.5),
    },
    outputs={"k_app": ("(wppm^(1-n))/hr", "Apparent nth-order plug-flow rate constant")},
    reference="Integrated plug-flow rate law: k = LHSV/(n-1) [Cp^(1-n) - Cf^(1-n)]",
)
def apparent_rate_constant(c_in, c_out, lhsv, order=1.5):
    """Apparent nth-order rate constant for plug flow."""
    return _k_nth(c_in, c_out, lhsv, order)


@equation(
    name="normalized_wabt",
    category="kinetics",
    inputs={
        "wabt": ("degF", "Actual WABT"),
        "lhsv": ("1/hr", "Actual LHSV"),
        "c_in": ("wppm", "Actual feed S (or N)"),
        "c_out": ("wppm", "Actual product S (or N)"),
        "h2_pp": ("psia", "Actual H2 partial pressure"),
        "lhsv_ref": ("1/hr", "Reference LHSV"),
        "c_in_ref": ("wppm", "Reference feed S"),
        "c_out_ref": ("wppm", "Reference product S"),
        "h2_pp_ref": ("psia", "Reference H2 partial pressure"),
        "order": ("", "Reaction order", 1.5),
        "ea_kj_mol": ("kJ/mol", "Activation energy", 110.0),
        "h2_exponent": ("", "H2 partial pressure exponent", 0.5),
    },
    outputs={"wabt_norm": ("degF", "WABT normalized to reference conditions")},
    reference="Arrhenius + nth-order plug-flow normalization; "
              "1/Tn = 1/Ta - (R/Ea) ln[(k_req/k_obs)(Pa/Pref)^alpha]",
)
def normalized_wabt(wabt, lhsv, c_in, c_out, h2_pp, lhsv_ref, c_in_ref, c_out_ref, h2_pp_ref,
                    order=1.5, ea_kj_mol=110.0, h2_exponent=0.5):
    """Temperature that would be needed at reference feed/rate/product/H2PP conditions.

    Removes the effect of feed rate, feed quality, product target and H2PP from the WABT
    trend so that what remains reflects catalyst deactivation."""
    k_obs = _k_nth(c_in, c_out, lhsv, order)
    k_req = _k_nth(c_in_ref, c_out_ref, lhsv_ref, order)
    ratio = (k_req / k_obs) * (h2_pp / h2_pp_ref) ** h2_exponent
    inv_t = 1.0 / f_to_k(wabt) - (R_KJ / ea_kj_mol) * np.log(ratio)
    return k_to_f(1.0 / inv_t)


@equation(
    name="required_temperature_change",
    category="kinetics",
    inputs={
        "k_ratio": ("", "Required rate constant / current rate constant"),
        "wabt": ("degF", "Current WABT"),
        "ea_kj_mol": ("kJ/mol", "Activation energy", 110.0),
    },
    outputs={"delta_t": ("degF", "WABT change needed")},
)
def required_temperature_change(k_ratio, wabt, ea_kj_mol=110.0):
    """WABT change needed to multiply the rate constant by k_ratio (Arrhenius)."""
    inv_t = 1.0 / f_to_k(wabt) - (R_KJ / ea_kj_mol) * np.log(k_ratio)
    return k_to_f(1.0 / inv_t) - wabt


# ---------------------------------------------------------------------------
# Catalyst life, balances, properties
# ---------------------------------------------------------------------------


@equation(
    name="catalyst_life",
    category="catalyst",
    inputs={
        "cumulative_feed_bbl": ("bbl", "Cumulative feed processed since start of run"),
        "catalyst_mass_lb": ("lb", "Catalyst mass"),
    },
    outputs={"life_bbl_lb": ("bbl/lb", "Catalyst life")},
)
def catalyst_life(cumulative_feed_bbl, catalyst_mass_lb):
    """Catalyst life as barrels processed per pound of catalyst."""
    return cumulative_feed_bbl / catalyst_mass_lb


@equation(
    name="mass_balance_closure",
    category="balance",
    inputs={"mass_in": ("lb/hr", "Inlet mass flows"), "mass_out": ("lb/hr", "Outlet mass flows")},
    outputs={"closure_pct": ("%", "Mass balance closure (out/in)")},
    list_inputs=("mass_in", "mass_out"),
)
def mass_balance_closure(mass_in, mass_out):
    """Mass balance closure = sum(out)/sum(in) x 100."""
    return sum(np.asarray(x, dtype=float) for x in mass_out) / sum(
        np.asarray(x, dtype=float) for x in mass_in) * 100.0


@equation(
    name="liquid_yield",
    category="balance",
    inputs={
        "product_bpd": ("BPD", "Product rate"),
        "feed_bpd": ("BPD", "Feed rate"),
        "product_sg": ("", "Product SG (for wt% yield)", 0.0),
        "feed_sg": ("", "Feed SG (for wt% yield)", 0.0),
    },
    outputs={"yield_vol_pct": ("vol%", "Volumetric yield"), "yield_wt_pct": ("wt%", "Mass yield")},
)
def liquid_yield(product_bpd, feed_bpd, product_sg=0.0, feed_sg=0.0):
    """Product yield on feed, volume and (if SGs given) mass basis."""
    vol = product_bpd / feed_bpd * 100.0
    wt = vol * product_sg / feed_sg if np.all(np.asarray(feed_sg) > 0) else np.nan * vol
    return {"yield_vol_pct": vol, "yield_wt_pct": wt}


@equation(
    name="api_gravity",
    category="properties",
    inputs={"sg": ("", "Specific gravity 60/60F")},
    outputs={"api": ("degAPI", "API gravity")},
)
def api_gravity(sg):
    """API gravity from specific gravity."""
    return 141.5 / sg - 131.5


@equation(
    name="specific_gravity",
    category="properties",
    inputs={"api": ("degAPI", "API gravity")},
    outputs={"sg": ("", "Specific gravity 60/60F")},
)
def specific_gravity(api):
    """Specific gravity from API gravity."""
    return 141.5 / (api + 131.5)


@equation(
    name="cetane_index_d4737",
    category="properties",
    inputs={
        "t10_f": ("degF", "D86 10% recovered"),
        "t50_f": ("degF", "D86 50% recovered"),
        "t90_f": ("degF", "D86 90% recovered"),
        "density_15c_g_ml": ("g/mL", "Density at 15 C"),
    },
    outputs={"cetane_index": ("", "Calculated cetane index (four-variable)")},
    reference="ASTM D4737 Procedure A",
)
def cetane_index_d4737(t10_f, t50_f, t90_f, density_15c_g_ml):
    """Four-variable calculated cetane index (ASTM D4737 Procedure A)."""
    d = density_15c_g_ml
    b = np.exp(-3.5 * (d - 0.85)) - 1.0
    t10n = f_to_c(t10_f) - 215.0
    t50n = f_to_c(t50_f) - 260.0
    t90n = f_to_c(t90_f) - 310.0
    return (45.2 + 0.0892 * t10n + (0.131 + 0.901 * b) * t50n
            + (0.0523 - 0.420 * b) * t90n + 0.00049 * (t10n ** 2 - t90n ** 2)
            + 107.0 * b + 60.0 * b ** 2)


@equation(
    name="pressure_drop_normalized",
    category="hydraulics",
    inputs={
        "dp_psi": ("psi", "Measured reactor/bed pressure drop"),
        "flow": ("any", "Actual flow (feed BPD or total mass flow)"),
        "flow_ref": ("any", "Reference flow, same units"),
        "exponent": ("", "Flow exponent (about 1.8-2.0 turbulent)", 1.9),
    },
    outputs={"dp_norm_psi": ("psi", "Pressure drop normalized to reference flow")},
)
def pressure_drop_normalized(dp_psi, flow, flow_ref, exponent=1.9):
    """Normalize pressure drop to a reference throughput to track fouling."""
    return dp_psi * (flow_ref / flow) ** exponent


@equation(
    name="sulfur_removed_lb_day",
    category="balance",
    inputs={
        "feed_rate_bpd": ("BPD", "Feed rate"),
        "feed_sg": ("", "Feed SG"),
        "feed_s_wppm": ("wppm", "Feed sulfur"),
        "product_s_wppm": ("wppm", "Product sulfur"),
    },
    outputs={"s_removed_lb_day": ("lb/day", "Sulfur removed"),
             "h2s_produced_lb_day": ("lb/day", "H2S produced")},
)
def sulfur_removed_lb_day(feed_rate_bpd, feed_sg, feed_s_wppm, product_s_wppm):
    """Sulfur removed and H2S formed (product mass assumed equal to feed mass)."""
    feed_lb = feed_rate_bpd * feed_sg * LB_PER_BBL_WATER
    s = feed_lb * (feed_s_wppm - product_s_wppm) * 1e-6
    return {"s_removed_lb_day": s, "h2s_produced_lb_day": s * 34.08 / 32.06}
