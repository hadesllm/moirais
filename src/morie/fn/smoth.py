"""Epidemic smoothing (Savitzky-Golay / LOESS for epidemiology)."""

from __future__ import annotations

from typing import Any

import numpy as np


def epidemic_smooth(
    incidence: np.ndarray,
    *,
    method: str = "savgol",
    window: int = 7,
    polyorder: int = 2,
    frac: float = 0.3,
) -> dict[str, Any]:
    """Smooth an epidemic curve using Savitzky-Golay or LOESS.

    Parameters
    ----------
    incidence : array_like
        Daily incidence counts.
    method : str, default "savgol"
        "savgol" for Savitzky-Golay or "loess" for local regression.
    window : int, default 7
        Window length for Savitzky-Golay (must be odd).
    polyorder : int, default 2
        Polynomial order for Savitzky-Golay.
    frac : float, default 0.3
        Fraction of data used in LOESS local neighborhood.

    Returns
    -------
    dict
        Keys: 'smoothed', 'residuals', 'method'.

    References
    ----------
    Savitzky, A. & Golay, M. J. E. (1964). Smoothing and
    differentiation of data by simplified least squares procedures.
    Analytical Chemistry, 36(8), 1627-1639.

    Cleveland, W. S. (1979). Robust locally weighted regression and
    smoothing scatterplots. Journal of the American Statistical
    Association, 74(368), 829-836.
    """
    inc = np.asarray(incidence, dtype=float)
    if inc.ndim != 1 or inc.size < 3:
        raise ValueError("incidence must be 1-D with >= 3 points.")

    if method == "savgol":
        from scipy.signal import savgol_filter
        if window % 2 == 0:
            window += 1
        window = min(window, len(inc))
        if window % 2 == 0:
            window -= 1
        if polyorder >= window:
            polyorder = window - 1
        smoothed = savgol_filter(inc, window, polyorder)
        smoothed = np.maximum(smoothed, 0)

    elif method == "loess":
        n = len(inc)
        t = np.arange(n, dtype=float)
        smoothed = np.empty(n)
        h = max(int(np.ceil(frac * n)), polyorder + 1)

        for i in range(n):
            dists = np.abs(t - t[i])
            idx = np.argsort(dists)[:h]
            max_d = dists[idx[-1]] + 1e-10
            w = (1 - (dists[idx] / max_d) ** 3) ** 3

            X = np.column_stack([t[idx] ** p for p in range(polyorder + 1)])
            W = np.diag(w)
            try:
                beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ inc[idx])
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ inc[idx], rcond=None)[0]
            x_i = np.array([t[i] ** p for p in range(polyorder + 1)])
            smoothed[i] = x_i @ beta

        smoothed = np.maximum(smoothed, 0)
    else:
        raise ValueError("method must be 'savgol' or 'loess'.")

    residuals = inc - smoothed

    return {
        "smoothed": smoothed,
        "residuals": residuals,
        "method": method,
    }


smoth = epidemic_smooth


def cheatsheet() -> str:
    return "epidemic_smooth({}) -> Smooth epidemic curve (SavGol/LOESS)."
