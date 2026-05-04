"""ARIMA-based forecasting of word-shift-score trajectories.

Given a time series of shift scores per word (length T-1 pairs), forecast
the next H steps. We try pmdarima.auto_arima if available, then fall back
to statsmodels.ARIMA with a fixed order, then to a naive last-value
forecast for very short series.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import pmdarima as pm  # noqa: F401

    _HAS_PMDARIMA = True
except Exception:  # pragma: no cover
    _HAS_PMDARIMA = False

try:
    from statsmodels.tsa.arima.model import ARIMA

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAS_STATSMODELS = False


def forecast_series(
    series: np.ndarray,
    horizon: int = 1,
    order: tuple[int, int, int] = (1, 1, 1),
    min_points: int = 8,
) -> np.ndarray:
    """Return the next `horizon` predicted values for `series`."""
    series = np.asarray(series, dtype=np.float64)
    if len(series) < min_points:
        # Naive last-value forecast when too few points.
        last = series[-1] if len(series) else 0.0
        return np.full(horizon, last, dtype=np.float64)
    if _HAS_PMDARIMA:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = pm.auto_arima(series, seasonal=False, error_action="ignore", suppress_warnings=True)
                fc = model.predict(n_periods=horizon)
                return np.asarray(fc, dtype=np.float64)
        except Exception:
            pass
    if _HAS_STATSMODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=order).fit()
                fc = model.forecast(steps=horizon)
                return np.asarray(fc, dtype=np.float64)
        except Exception:
            pass
    # Ultimate fallback.
    return np.full(horizon, series[-1], dtype=np.float64)


def rolling_forecast_eval(
    series: np.ndarray,
    min_train: int = 3,
    order: tuple[int, int, int] = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window 1-step forecast; returns (predictions, actuals)."""
    series = np.asarray(series, dtype=np.float64)
    preds = []
    actuals = []
    for k in range(min_train, len(series)):
        fc = forecast_series(series[:k], horizon=1, order=order, min_points=1)
        preds.append(float(fc[0]))
        actuals.append(float(series[k]))
    return np.asarray(preds), np.asarray(actuals)
