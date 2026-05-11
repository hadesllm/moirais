# morie.fn — function file (hadesllm/morie)
"""ACF / PACF plot visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_acf(
    values: Sequence[float],
    *,
    nlags: int = 40,
    pacf: bool = False,
    ax: Any | None = None,
) -> Any:
    """
    Autocorrelation or partial autocorrelation bar plot with approximate
    95 % confidence bands.

    ACF is computed via the standard definition.  When ``pacf=True``,
    the Yule-Walker method (via ``numpy.linalg.solve``) is used for
    the partial autocorrelation.

    :param values: Time series values.
    :param nlags: Maximum lag. Default 40.
    :param pacf: If True, plot PACF instead of ACF.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (2015). *Time Series
        Analysis: Forecasting and Control* (5th ed.). Wiley.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_acf requires matplotlib. Install via: pip install matplotlib")
        return None

    x = np.asarray(values, dtype=float)
    n = len(x)
    nlags = min(nlags, n - 1)

    if pacf:
        acf_vals = _compute_acf(x, nlags)
        coeffs = _compute_pacf(acf_vals, nlags)
        title = "PACF"
    else:
        coeffs = _compute_acf(x, nlags)
        title = "ACF"

    lags = np.arange(len(coeffs))
    ci = 1.96 / np.sqrt(n)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.bar(lags, coeffs, width=0.3, color="steelblue", edgecolor="black")
    ax.axhline(ci, linestyle="--", color="red", linewidth=0.7)
    ax.axhline(-ci, linestyle="--", color="red", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel(title)
    ax.set_title(title)
    return fig


def _compute_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute ACF up to *nlags*."""
    xm = x - x.mean()
    var = np.dot(xm, xm)
    if var == 0:
        return np.zeros(nlags + 1)
    acf = np.array([np.dot(xm[: len(xm) - k], xm[k:]) / var for k in range(nlags + 1)])
    return acf


def _compute_pacf(acf: np.ndarray, nlags: int) -> np.ndarray:
    """Derive PACF from ACF via Yule-Walker (Durbin-Levinson)."""
    pacf_vals = [1.0]
    for k in range(1, nlags + 1):
        r = acf[1 : k + 1]
        if k == 1:
            pacf_vals.append(r[0])
            continue
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = acf[abs(i - j)]
        try:
            phi = np.linalg.solve(R, r)
            pacf_vals.append(phi[-1])
        except np.linalg.LinAlgError:
            pacf_vals.append(0.0)
    return np.array(pacf_vals)


def cheatsheet() -> str:
    return "holo_acf({}) -> ACF / PACF plot visualization."
