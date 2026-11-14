"""Time-series operations for historical analysis: applying equations over datasets,
steady-state detection, filtering, resampling, catalyst age, window statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hydromcp.core.equations import REGISTRY, to_jsonable


def _resolve(df: pd.DataFrame, ref: Any):
    """A column_map value may be a column name, a list of column names, or a constant."""
    if isinstance(ref, str):
        if ref not in df.columns:
            raise KeyError(f"Column '{ref}' not in dataset. Columns: {list(df.columns)}")
        return df[ref]
    if isinstance(ref, (list, tuple)):
        return [_resolve(df, r) for r in ref]
    return ref  # numeric constant


def apply_equation(df: pd.DataFrame, equation_name: str, column_map: dict[str, Any],
                   output_prefix: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate an equation row-by-row over a DataFrame.

    column_map: {equation_input: column name | [column names] | constant}.
    Returns (new df with output columns appended, list of new column names)."""
    eq = REGISTRY.get(equation_name)
    kwargs = {name: _resolve(df, ref) for name, ref in column_map.items()}
    result = eq.evaluate(**kwargs)
    out = df.copy()
    new_cols = []
    for key, val in result.items():
        col = f"{output_prefix}{key}" if output_prefix else key
        if np.isscalar(val):
            val = pd.Series(val, index=df.index)
        out[col] = pd.Series(np.asarray(val, dtype=float), index=df.index)
        new_cols.append(col)
    return out, new_cols


def describe(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    desc = df[cols].describe().T
    desc["missing"] = df[cols].isna().sum()
    return {c: {k: to_jsonable(v) for k, v in row.items()} for c, row in desc.iterrows()}


def window_stats(df: pd.DataFrame, columns: list[str], start: str | None = None,
                 end: str | None = None) -> dict:
    sub = df.loc[start:end] if isinstance(df.index, pd.DatetimeIndex) else df
    return {
        "rows": len(sub),
        "mean": to_jsonable(sub[columns].mean().to_dict()),
        "median": to_jsonable(sub[columns].median().to_dict()),
        "std": to_jsonable(sub[columns].std().to_dict()),
    }


def resample(df: pd.DataFrame, rule: str, how: str = "mean") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataset needs a datetime index to resample")
    numeric = df.select_dtypes("number")
    return getattr(numeric.resample(rule), how)()


def filter_outliers(df: pd.DataFrame, columns: list[str], window: int = 25,
                    n_mad: float = 4.0) -> tuple[pd.DataFrame, dict]:
    """Replace points further than n_mad scaled MADs from a rolling median with NaN."""
    out = df.copy()
    removed = {}
    for c in columns:
        s = out[c]
        med = s.rolling(window, center=True, min_periods=1).median()
        mad = (s - med).abs().rolling(window, center=True, min_periods=1).median() * 1.4826
        mask = (s - med).abs() > n_mad * mad.replace(0, np.nan)
        removed[c] = int(mask.sum())
        out.loc[mask, c] = np.nan
    return out, removed


def detect_steady_state(df: pd.DataFrame, columns: list[str], window: str | int = "24h",
                        tolerances: dict[str, float] | None = None, rel_tol: float = 0.02,
                        min_duration: str | None = None) -> tuple[pd.Series, list[dict]]:
    """Flag steady-state rows: rolling std of each column within tolerance.

    tolerances: absolute std limit per column (e.g. {"feed_rate": 500, "wabt": 2}).
    Columns without an absolute tolerance use rel_tol * |rolling mean|.
    Returns (boolean series, list of contiguous steady windows)."""
    tolerances = tolerances or {}
    flag = pd.Series(True, index=df.index)
    for c in columns:
        roll = df[c].rolling(window, min_periods=2)
        std = roll.std()
        lim = tolerances.get(c)
        limit = lim if lim is not None else rel_tol * roll.mean().abs()
        flag &= (std <= limit).fillna(False)
    windows = []
    if flag.any():
        grp = (flag != flag.shift()).cumsum()
        for _, seg in flag[flag].groupby(grp[flag]):
            w = {"start": str(seg.index[0]), "end": str(seg.index[-1]), "rows": len(seg)}
            if isinstance(df.index, pd.DatetimeIndex):
                dur = seg.index[-1] - seg.index[0]
                if min_duration and dur < pd.Timedelta(min_duration):
                    continue
                w["duration"] = str(dur)
            windows.append(w)
    return flag, windows


def cumulative_feed(df: pd.DataFrame, feed_rate_col: str, start: str | None = None) -> pd.Series:
    """Cumulative barrels processed (trapezoidal integration of BPD over time)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataset needs a datetime index")
    s = df[feed_rate_col].astype(float)
    if start:
        s = s.loc[start:]
    days = (s.index - s.index[0]).total_seconds() / 86400.0
    rate = s.interpolate(limit_direction="both").to_numpy()
    increments = np.concatenate([[0.0], 0.5 * (rate[1:] + rate[:-1]) * np.diff(days)])
    return pd.Series(np.cumsum(increments), index=s.index, name="cumulative_feed_bbl").reindex(
        df.index)


def days_on_stream(df: pd.DataFrame, start: str | None = None) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataset needs a datetime index")
    t0 = pd.Timestamp(start) if start else df.index[0]
    return pd.Series((df.index - t0).total_seconds() / 86400.0, index=df.index,
                     name="days_on_stream")


def correlate(df: pd.DataFrame, columns: list[str]) -> dict:
    return to_jsonable(df[columns].corr().round(4).to_dict())
