"""Weighted absolute percentage error (WAPE) and scaled metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wapef(y_true, y_pred, weights=None):
    """Weighted absolute percentage error.

    Parameters
    ----------
    y_true : array-like
        Actual values (n,).
    y_pred : array-like
        Predicted values (n,).
    weights : array-like, optional
        Observation weights (n,). If None, uniform weights.

    Returns
    -------
    DescriptiveResult
        Fields: wape (float), mae, mape, wmape.

    Notes
    -----
    WAPE = sum(w_i * |y_i - yhat_i|) / sum(w_i * |y_i|)

    Scaled by sum of absolute values for interpretability.

    References
    ----------
    Goodwin, P., & Lawton, R. (1999). On the asymmetry of the symmetric MAPE.
    International Journal of Forecasting, 15(4), 405-408.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    n = len(y_true)

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if len(weights) != n:
            raise ValueError("weights must have same length as y_true")

    # Ensure weights are positive
    weights = np.abs(weights) / np.sum(np.abs(weights))

    # Absolute error
    abs_error = np.abs(y_true - y_pred)

    # Absolute percentage error
    ape = np.where(np.abs(y_true) > 1e-10,
                   100 * abs_error / np.abs(y_true),
                   0.0)

    # Weighted metrics
    mae = np.sum(weights * abs_error)
    wmape = np.sum(weights * np.abs(y_true - y_pred)) / (np.sum(weights * np.abs(y_true)) + 1e-10)
    mape = np.mean(ape)
    wape = np.sum(weights * abs_error) / (np.sum(weights * np.abs(y_true)) + 1e-10)

    return DescriptiveResult(
        name="wape_metrics",
        value=float(wape),
        extra={
            "wape": float(wape),
            "wmape": float(wmape),
            "mape": float(mape),
            "mae": float(mae),
        },
    )


wape_metrics = wapef


def cheatsheet() -> str:
    return "wapef(y_true, y_pred, weights=None) -> Weighted absolute percentage error"
