"""Generic time-series forecasting of process variables / yields."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def forecast_series(s: pd.Series, horizon: int, freq: str = "D", method: str = "holt",
                    damped: bool = True, seasonal_periods: int | None = None,
                    confidence: float = 0.9) -> dict:
    """Forecast a series `horizon` steps of `freq` ahead.

    method: 'holt' (Holt / Holt-Winters exponential smoothing), 'arima' (ARIMA(1,1,1)),
            'linear' (OLS trend with prediction interval)."""
    s = s.dropna().astype(float)
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.resample(freq).mean().interpolate(limit_direction="both")
    if len(s) < 10:
        raise ValueError("Need at least 10 points after resampling")
    alpha = 1 - confidence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "holt":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                s, trend="add", damped_trend=damped,
                seasonal="add" if seasonal_periods else None,
                seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
            mean = model.forecast(horizon)
            resid_std = float(np.std(model.resid, ddof=1))
            steps = np.arange(1, horizon + 1)
            from scipy import stats

            z = stats.norm.ppf(1 - alpha / 2)
            lower, upper = mean - z * resid_std * np.sqrt(steps), mean + z * resid_std * np.sqrt(steps)
            info = {"aic": float(model.aic)}
        elif method == "arima":
            from statsmodels.tsa.arima.model import ARIMA

            res = ARIMA(s, order=(1, 1, 1)).fit()
            fc = res.get_forecast(horizon)
            mean = fc.predicted_mean
            ci = fc.conf_int(alpha=alpha)
            lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
            info = {"aic": float(res.aic)}
        elif method == "linear":
            import statsmodels.api as sm

            t = np.arange(len(s), dtype=float)
            res = sm.OLS(s.to_numpy(), sm.add_constant(t)).fit()
            tf = np.arange(len(s), len(s) + horizon, dtype=float)
            pred = res.get_prediction(sm.add_constant(tf, has_constant="add"))
            frame = pred.summary_frame(alpha=alpha)
            idx = (pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
                   if isinstance(s.index, pd.DatetimeIndex) else tf)
            mean = pd.Series(frame["mean"].to_numpy(), index=idx)
            lower = pd.Series(frame["obs_ci_lower"].to_numpy(), index=idx)
            upper = pd.Series(frame["obs_ci_upper"].to_numpy(), index=idx)
            info = {"slope_per_step": float(res.params[1]), "r2": float(res.rsquared)}
        else:
            raise ValueError("method must be holt, arima or linear")
    out = pd.DataFrame({"forecast": np.asarray(mean), "lower": np.asarray(lower),
                        "upper": np.asarray(upper)}, index=mean.index)
    return {"frame": out, "info": info, "last_value": float(s.iloc[-1]),
            "last_time": str(s.index[-1])}
