# morie.fn -- function file (hadesllm/morie)
"""Generalized additive model (B-spline basis + OLS). 'Size matters not.'"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from ._containers import DescriptiveResult


def _bspline_basis(x: np.ndarray, n_splines: int = 10, degree: int = 3) -> np.ndarray:
    """Build a B-spline design matrix."""
    n_internal = n_splines - degree - 1
    lo, hi = float(x.min()), float(x.max())
    internal_knots = np.linspace(lo, hi, n_internal + 2)[1:-1]
    knots = np.concatenate(
        [
            np.full(degree + 1, lo),
            internal_knots,
            np.full(degree + 1, hi),
        ]
    )
    n_basis = len(knots) - degree - 1
    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=False)
        B[:, i] = spl(x)
    B = np.nan_to_num(B, nan=0.0)
    return B


def fit_gam(
    x: np.ndarray,
    y: np.ndarray,
    n_splines: int = 10,
) -> DescriptiveResult:
    """Fit a GAM using B-spline basis expansion + OLS.

    Parameters
    ----------
    x : ndarray
        Predictor (1D).
    y : ndarray
        Response.
    n_splines : int, default 10
        Number of spline basis functions.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    B = _bspline_basis(x, n_splines=n_splines)
    B = np.column_stack([np.ones(len(x)), B])
    coeffs, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    predicted = B @ coeffs
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="GAM (B-spline + OLS)",
        value=r_squared,
        extra={
            "coefficients": coeffs,
            "predicted": predicted,
            "residuals": residuals,
            "r_squared": r_squared,
            "n_splines": n_splines,
            "n": len(x),
        },
    )


gam = fit_gam


def cheatsheet() -> str:
    return "_bspline_basis({}) -> Generalized additive model (B-spline basis + OLS). 'Size mat"
