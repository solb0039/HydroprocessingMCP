"""Section 3 MCP tools: deactivation trends, end-of-run, variable forecasts, what-if."""

from __future__ import annotations

import json
from typing import Literal

from fastmcp import FastMCP

from hydromcp.core.equations import to_jsonable
from hydromcp.core.state import STORE
from hydromcp.forecast import deactivation as deact
from hydromcp.forecast.scenarios import what_if
from hydromcp.forecast.timeseries import forecast_series

mcp = FastMCP("forecast")

Kind = Literal["linear", "quadratic", "piecewise", "exponential"]


def _xy(dataset_id, y_column, x_column, start, end, steady_only_column):
    df = STORE.dataset(dataset_id).df
    if start or end:
        df = df.loc[start:end]
    if steady_only_column:
        df = df[df[steady_only_column].astype(bool)]
    if x_column not in df.columns:
        raise KeyError(f"Column '{x_column}' not found; run add_catalyst_age first or pass x_column")
    sub = df[[x_column, y_column]].dropna()
    t0 = None
    full = STORE.dataset(dataset_id).df
    if x_column == "days_on_stream" and len(full):
        first = full[x_column].dropna()
        if len(first):
            import pandas as pd

            t0 = str(first.index[0] - pd.Timedelta(days=float(first.iloc[0])))
    return sub[x_column].to_numpy(), sub[y_column].to_numpy(), t0


@mcp.tool
def fit_deactivation(dataset_id: str, y_column: str = "wabt_norm",
                     x_column: str = "days_on_stream", kind: Kind = "linear",
                     start: str | None = None, end: str | None = None,
                     steady_only_column: str | None = None, min_x: float | None = None) -> dict:
    """Fit a deactivation trend of (normalized) WABT vs catalyst age (days_on_stream or
    catalyst_life_bbl_lb). Optionally use only rows flagged steady. Returns a model_id."""
    x, y, t0 = _xy(dataset_id, y_column, x_column, start, end, steady_only_column)
    model = deact.fit_deactivation(x, y, kind, x_column, t0, min_x)
    mid = STORE.add_model(model, "deact")
    return {"model_id": mid, **to_jsonable(model.summary())}


@mcp.tool
def compare_deactivation_models(dataset_id: str, y_column: str = "wabt_norm",
                                x_column: str = "days_on_stream", start: str | None = None,
                                end: str | None = None, steady_only_column: str | None = None,
                                min_x: float | None = None) -> dict:
    """Fit all deactivation model types and rank them by RMSE (each gets a model_id)."""
    x, y, t0 = _xy(dataset_id, y_column, x_column, start, end, steady_only_column)
    out = []
    for kind in ("linear", "quadratic", "piecewise", "exponential"):
        try:
            m = deact.fit_deactivation(x, y, kind, x_column, t0, min_x)
            out.append({"model_id": STORE.add_model(m, "deact"), "kind": kind, "r2": m.r2,
                        "rmse": m.rmse, "current_slope_per_x": m.current_rate()})
        except Exception as exc:
            out.append({"kind": kind, "error": str(exc)})
    ranked = sorted([o for o in out if "rmse" in o], key=lambda o: o["rmse"])
    return to_jsonable({"models": out, "best": ranked[0]["model_id"] if ranked else None,
                        "note": "Prefer the simplest model with similar RMSE for extrapolation"})


@mcp.tool
def predict_end_of_run(model_id: str, max_wabt_f: float, x_per_day: float | None = None,
                       confidence: float = 0.9) -> dict:
    """Predict when the deactivation trend reaches the max (EOR) WABT.

    x_per_day converts the model's x to calendar days: 1.0 for days_on_stream (default),
    or feed_bpd / catalyst_lb when x is catalyst_life_bbl_lb."""
    model = STORE.model(model_id)
    if x_per_day is None and model.x_name == "days_on_stream":
        x_per_day = 1.0
    return to_jsonable(deact.predict_eor(model, max_wabt_f, x_per_day, confidence))


@mcp.tool
def forecast_variable(dataset_id: str, column: str, horizon: int = 90, freq: str = "D",
                      method: Literal["holt", "arima", "linear"] = "holt", damped: bool = True,
                      seasonal_periods: int | None = None, confidence: float = 0.9,
                      max_points: int = 60) -> dict:
    """Forecast a process variable or yield forward (horizon steps of freq) with an
    uncertainty band. Full forecast is stored as a new dataset; a sample is returned."""
    s = STORE.dataset(dataset_id).df[column]
    res = forecast_series(s, horizon, freq, method, damped, seasonal_periods, confidence)
    frame = res["frame"]
    entry = STORE.add_dataset(frame, f"forecast:{dataset_id}:{column}:{method}")
    step = max(1, len(frame) // max_points)
    sample = frame.iloc[::step]
    if frame.index[-1] not in sample.index:
        sample = frame.iloc[list(range(0, len(frame), step)) + [len(frame) - 1]]
    return {"forecast_dataset_id": entry.id, "method": method, "info": res["info"],
            "last_observed": {"time": res["last_time"], "value": res["last_value"]},
            "end_of_horizon": to_jsonable(frame.iloc[-1].to_dict()),
            "points": json.loads(sample.reset_index().to_json(orient="records",
                                                              date_format="iso"))}


@mcp.tool
def what_if_scenario(calibration_id: str, changes: dict, age_months: float | None = None,
                     product_sulfur_target_wppm: float | None = None,
                     proposal_id: str | None = None) -> dict:
    """Effect of changed operation on a calibrated unit: required WABT, H2 use and remaining
    cycle. changes are OperatingBasis fields, e.g. {'feed_rate_bpd': 42000,
    'feed_sulfur_wppm': 14000, 'h2_partial_pressure_psia': 850}."""
    return to_jsonable(what_if(STORE.calibration(calibration_id), changes, age_months,
                               product_sulfur_target_wppm, proposal_id))
