"""Generate a synthetic ULSD historian export (raw tag names) for demos and tests.

The data are produced from the kinetic model of the example unit ULSD-1 with its current
loading, so the analysis -> deactivation -> calibration workflow can be checked against
known answers (reference catalyst deactivation = 1.5 F/month).

    uv run python scripts/generate_sample_data.py [--days 420] [--freq 4h]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydromcp.core.library import get_library
from hydromcp.core.schemas import OperatingBasis
from hydromcp.proposals.predict import build_system, calibrate


def generate(days: int = 420, freq: str = "4h", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lib = get_library()
    unit = lib.unit("ULSD-1")
    loading = lib.proposal("ULSD-1:current")
    cal = calibrate(unit, loading, unit.design_basis, 0.0, lib)
    system, plan = build_system(loading, unit, lib)
    system.k_ref, system.h2pp_ref_psia = cal.k_ref, cal.h2pp_ref_psia

    idx = pd.date_range("2025-01-01", periods=int(days * 24 / pd.Timedelta(freq).total_seconds()
                                                  * 3600), freq=freq)
    n = len(idx)
    t_days = np.arange(n) * pd.Timedelta(freq).total_seconds() / 86400

    def walk(scale, smooth):
        x = np.cumsum(rng.normal(0, scale, n))
        return pd.Series(x).ewm(span=smooth).mean().to_numpy() - x.mean()

    feed = 35000 + walk(40, 30) + np.where((t_days > 120) & (t_days < 160), -4000, 0)
    feed = np.clip(feed + rng.normal(0, 150, n), 25000, 41000)
    feed_s = np.clip(12000 + walk(25, 60) + 1500 * np.sin(t_days / 45), 9000, 15500)
    feed_n = np.clip(250 + walk(1, 60), 150, 380)
    feed_sg = 0.865 + walk(0.00008, 60)
    h2pp = 900 + walk(0.5, 40)
    target = 8.0

    wabt = np.empty(n)
    for i in range(n):
        b = OperatingBasis(feed_rate_bpd=feed[i], feed_sg=feed_sg[i], feed_sulfur_wppm=feed_s[i],
                           h2_partial_pressure_psia=h2pp[i])
        wabt[i] = system.required_wabt("hds", feed_s[i], target, b, t_days[i] / 30.4375)
    wabt += rng.normal(0, 1.2, n)
    prod_s = np.clip(target * np.exp(rng.normal(0, 0.12, n)), 2, None)

    # bed temperatures consistent with catalyst-weighted WABT
    bed_lb = [sum(l.mass_lb for l in plan["_layers"] if l.bed_name == b and l.active)
              for b in ("bed1", "bed2")]
    w1 = bed_lb[0] / sum(bed_lb)
    rise1 = 28 + 6 * (feed_s - 12000) / 3000 + rng.normal(0, 1, n)
    rise2 = 18 + 4 * (feed_s - 12000) / 3000 + rng.normal(0, 1, n)
    wabt1 = wabt - 9 * (1 - w1)
    wabt2 = (wabt - w1 * wabt1) / (1 - w1)
    b1_in, b2_in = wabt1 - 2 * rise1 / 3, wabt2 - 2 * rise2 / 3

    y_h2 = np.clip(0.82 + walk(0.0005, 50), 0.75, 0.9)
    p_psig = h2pp / y_h2 - 14.696
    h2_cons = 310 * (feed_s / 12000) ** 0.6 + rng.normal(0, 8, n)
    purge = 1500 + rng.normal(0, 40, n)
    mu_purity = 0.97 + rng.normal(0, 0.003, n)
    makeup = (h2_cons * feed / 1000 + purge * y_h2 + 400) / mu_purity
    dp = 22 * (feed / 35000) ** 1.9 * (1 + 0.25 * (t_days / days) ** 2) + rng.normal(0, 0.4, n)

    df = pd.DataFrame({
        "Timestamp": idx,
        "11FI100.PV": feed.round(0), "11AI101_SG": feed_sg.round(4),
        "11AI102_S": feed_s.round(0), "11AI103_N": feed_n.round(0), "11AI201_S": prod_s.round(2),
        "11TI110.PV": b1_in.round(1), "11TI111.PV": (b1_in + rise1).round(1),
        "11TI120.PV": b2_in.round(1), "11TI121.PV": (b2_in + rise2).round(1),
        "11PI130.PV": p_psig.round(1), "11AI131_H2": (y_h2 * 100).round(2),
        "11FI140.PV": makeup.round(0), "11AI141_H2": (mu_purity * 100).round(2),
        "11FI150.PV": purge.round(0), "11PDI160.PV": dp.round(2),
    })
    # a few historian glitches
    bad = rng.choice(n, 12, replace=False)
    df.loc[bad[:6], "11TI111.PV"] = np.nan
    df["11AI201_S"] = df["11AI201_S"].astype(object)
    df.loc[bad[6:], "11AI201_S"] = "Bad Input"
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=420)
    ap.add_argument("--freq", default="4h")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data/sample/synthetic_ulsd.csv"))
    a = ap.parse_args()
    out = generate(a.days, a.freq)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")
