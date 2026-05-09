# moirais.fn — function file (hadesllm/moirais)
"""Dose-response curve modeling (log-logistic)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ._containers import DescriptiveResult


def _log_logistic(x, b, c, d, e):
    """4-parameter log-logistic function."""
    return c + (d - c) / (1 + (x / e) ** b)


def dose_response_curve(
    doses: np.ndarray | list,
    responses: np.ndarray | list,
    *,
    model: str = "log_logistic",
) -> DescriptiveResult:
    """
    Fit a dose-response curve using a 4-parameter log-logistic model.

    .. math::

        f(x) = c + \\frac{d - c}{1 + (x/e)^b}

    Parameters
    ----------
    doses : array-like
        Dose levels (positive).
    responses : array-like
        Observed responses (proportions or means).
    model : str
        Currently only 'log_logistic'.

    Returns
    -------
    DescriptiveResult
        extra has 'params', 'ec50', 'residuals', 'r_squared'.

    References
    ----------
    Ritz, C. (2010). Toward a unified approach to dose-response
    modeling in ecotoxicology. *Environ Toxicol Chem*, 29(1), 220-229.
    """
    x = np.asarray(doses, dtype=float)
    y = np.asarray(responses, dtype=float)
    if len(x) != len(y):
        raise ValueError("doses and responses must match.")
    if len(x) < 4:
        raise ValueError("Need at least 4 data points.")
    if np.any(x <= 0):
        raise ValueError("Doses must be positive for log-logistic model.")

    b0 = -1.0
    c0 = float(np.min(y))
    d0 = float(np.max(y))
    e0 = float(np.median(x))

    popt, pcov = curve_fit(_log_logistic, x, y, p0=[b0, c0, d0, e0], maxfev=10000)
    y_pred = _log_logistic(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DescriptiveResult(
        name="dose_response",
        value=float(popt[3]),
        extra={
            "params": {"b": popt[0], "c": popt[1], "d": popt[2], "e": popt[3]},
            "ec50": float(popt[3]),
            "residuals": (y - y_pred).tolist(),
            "r_squared": float(r2),
            "model": model,
        },
    )


drcur = dose_response_curve


def cheatsheet() -> str:
    return "_log_logistic({}) -> Dose-response curve modeling (log-logistic)."
