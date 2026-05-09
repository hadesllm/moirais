"""Cubic spline regression. 'Knowledge itself is power. — Francis Bacon'

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ._containers import DescriptiveResult


def spline_regression(
    x: np.ndarray,
    y: np.ndarray,
    n_knots: int = 5,
) -> DescriptiveResult:
    """Cubic spline interpolation/regression.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    n_knots : int, default 5
        Number of knot points (evenly spaced quantiles).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    quantiles = np.linspace(0, 100, n_knots)
    knot_x = np.percentile(x_s, quantiles)
    knot_y = np.array(
        [
            np.mean(y_s[np.abs(x_s - kx) < (kx * 0.05 + 1e-10)])
            if np.any(np.abs(x_s - kx) < (kx * 0.05 + 1e-10))
            else np.interp(kx, x_s, y_s)
            for kx in knot_x
        ]
    )

    _, unique_idx = np.unique(knot_x, return_index=True)
    knot_x = knot_x[unique_idx]
    knot_y = knot_y[unique_idx]

    if len(knot_x) < 2:
        predicted = np.full_like(y, np.mean(y))
    else:
        cs = CubicSpline(knot_x, knot_y)
        predicted = cs(x)

    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DescriptiveResult(
        name="Cubic spline regression",
        value=r_squared,
        extra={
            "predicted": predicted,
            "residuals": residuals,
            "r_squared": r_squared,
            "n_knots": n_knots,
            "n": len(x),
        },
    )


spln = spline_regression


def cheatsheet() -> str:
    return "spline_regression({}) -> Cubic spline regression. 'An elegant weapon for a more civil"
