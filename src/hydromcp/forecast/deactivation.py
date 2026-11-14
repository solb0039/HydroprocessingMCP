"""Catalyst deactivation trend fitting and end-of-run (EOR) prediction.

Fit normalized WABT (y) against catalyst age x (days on stream or bbl/lb):
  linear     y = a + b x
  quadratic  y = a + b x + c x^2
  piecewise  two linear segments with fitted breakpoint (initial fast coking + stable)
  exponential y = a + b (1 - exp(-x/tau)) + c x  (fast initial deactivation then linear)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, curve_fit, minimize_scalar


@dataclass
class DeactivationModel:
    kind: str
    x_name: str
    params: dict
    r2: float
    rmse: float
    n: int
    x_last: float
    y_last_fit: float
    resid_std: float
    x_range: tuple[float, float]
    start: str | None = None  # timestamp of x=0 when x is days on stream
    extra: dict = field(default_factory=dict)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        p = self.params
        if self.kind == "linear":
            return p["a"] + p["b"] * x
        if self.kind == "quadratic":
            return p["a"] + p["b"] * x + p["c"] * x ** 2
        if self.kind == "piecewise":
            xb = p["breakpoint"]
            return np.where(x <= xb, p["a"] + p["b1"] * x,
                            p["a"] + p["b1"] * xb + p["b2"] * (x - xb))
        if self.kind == "exponential":
            return p["a"] + p["b"] * (1 - np.exp(-x / p["tau"])) + p["c"] * x
        raise ValueError(self.kind)

    def current_rate(self, dx: float = 1.0) -> float:
        """Slope at the last data point (y units per x unit)."""
        return float((self.predict(self.x_last + dx) - self.predict(self.x_last)) / dx)

    def summary(self) -> dict:
        return {"kind": self.kind, "x": self.x_name, "params": self.params, "r2": self.r2,
                "rmse": self.rmse, "n_points": self.n, "x_range": list(self.x_range),
                "current_slope_per_x": self.current_rate(), "start": self.start, **self.extra}


def _metrics(y, yhat, k):
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    dof = max(len(y) - k, 1)
    return 1 - ss_res / ss_tot, float(np.sqrt(ss_res / len(y))), float(np.sqrt(ss_res / dof))


def fit_deactivation(x, y, kind: str = "linear", x_name: str = "days_on_stream",
                     start: str | None = None, min_x: float | None = None) -> DeactivationModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if min_x is not None:
        m &= x >= min_x
    x, y = x[m], y[m]
    if len(x) < 5:
        raise ValueError("Need at least 5 valid points to fit a deactivation trend")
    extra = {}
    if kind == "linear":
        res = stats.linregress(x, y)
        params = {"a": res.intercept, "b": res.slope}
        extra["slope_stderr"] = res.stderr
        k = 2
    elif kind == "quadratic":
        c, b, a = np.polyfit(x, y, 2)
        params, k = {"a": a, "b": b, "c": c}, 3
    elif kind == "piecewise":
        def sse(xb):
            X = np.column_stack([np.ones_like(x), np.minimum(x, xb), np.maximum(x - xb, 0)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            return float(np.sum((X @ coef - y) ** 2)), coef
        lo, hi = np.percentile(x, 5), np.percentile(x, 80)
        opt = minimize_scalar(lambda xb: sse(xb)[0], bounds=(lo, hi), method="bounded")
        _, coef = sse(opt.x)
        params, k = {"a": coef[0], "b1": coef[1], "b2": coef[2], "breakpoint": float(opt.x)}, 4
    elif kind == "exponential":
        f = lambda xx, a, b, tau, c: a + b * (1 - np.exp(-xx / tau)) + c * xx
        span = max(x.max() - x.min(), 1.0)
        p0 = [y[: max(3, len(y) // 20)].mean(), max(y.max() - y.min(), 1.0) / 2, span / 10, 0.0]
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=([-np.inf, 0, 1e-6, -np.inf], np.inf),
                            maxfev=20000)
        params, k = dict(zip(["a", "b", "tau", "c"], popt)), 4
    else:
        raise ValueError("kind must be linear, quadratic, piecewise or exponential")
    params = {kk: float(v) for kk, v in params.items()}
    model = DeactivationModel(kind=kind, x_name=x_name, params=params, r2=0, rmse=0, n=len(x),
                              x_last=float(x.max()), y_last_fit=0, resid_std=0,
                              x_range=(float(x.min()), float(x.max())), start=start, extra=extra)
    yhat = model.predict(x)
    model.r2, model.rmse, model.resid_std = _metrics(y, yhat, k)
    model.y_last_fit = float(model.predict(model.x_last))
    return model


def predict_eor(model: DeactivationModel, max_wabt_f: float, x_per_day: float | None = None,
                confidence: float = 0.9, horizon_factor: float = 20.0) -> dict:
    """x at which the fitted trend reaches max WABT, with a band from residual scatter.

    x_per_day converts x to calendar days (1 for days_on_stream; for bbl/lb pass
    feed_bpd / catalyst_lb). Extrapolation uses the fitted curve beyond the data."""
    z = stats.norm.ppf(0.5 + confidence / 2)
    band = z * model.resid_std
    x_hi = model.x_last + horizon_factor * max(model.x_range[1] - model.x_range[0], 1.0)

    def cross(offset):
        g = lambda xx: float(model.predict(xx)) + offset - max_wabt_f
        if g(model.x_last) >= 0:
            return model.x_last
        if g(x_hi) < 0:
            return None
        return brentq(g, model.x_last, x_hi)

    x_eor = cross(0.0)
    x_early, x_late = cross(+band), cross(-band)
    out = {"max_wabt_f": max_wabt_f, "current_fit_wabt_f": model.y_last_fit,
           "current_x": model.x_last, "x_name": model.x_name,
           "current_rate_per_x": model.current_rate(),
           "eor_x": x_eor, "eor_x_early": x_early, "eor_x_late": x_late,
           "confidence": confidence}
    if x_eor is None:
        out["note"] = "Trend does not reach max WABT within the extrapolation horizon"
    if x_per_day:
        to_days = lambda v: None if v is None else (v - model.x_last) / x_per_day
        out["days_remaining"] = to_days(x_eor)
        out["days_remaining_early"] = to_days(x_early)
        out["days_remaining_late"] = to_days(x_late)
        out["rate_f_per_month"] = model.current_rate() * x_per_day * 30.4375
        if model.start and model.x_name == "days_on_stream" and x_eor is not None:
            out["eor_date"] = str((pd.Timestamp(model.start) + pd.Timedelta(days=x_eor)).date())
    return out
