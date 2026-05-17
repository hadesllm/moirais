"""Compute aerodynamic lift-drag polar and performance metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lift_drag_polar(
    cl: np.ndarray,
    cd: np.ndarray,
    *,
    fit_order: int = 2,
) -> DescriptiveResult:
    """Compute aerodynamic lift-drag polar and performance metrics.

    Fits Cd = Cd0 + K * Cl^2 (parabolic drag polar) and computes
    maximum L/D ratio and best glide speed parameters.

    Parameters
    ----------
    cl : array-like
        Lift coefficient data.
    cd : array-like
        Drag coefficient data.
    fit_order : int
        Polynomial order for Cd vs Cl^2 fit (default 2 = parabolic).

    Returns
    -------
    DescriptiveResult
        With ``value`` = max L/D ratio and ``extra`` containing
        Cd0, K, and optimal Cl.
    """
    cl = np.asarray(cl, dtype=float).ravel()
    cd = np.asarray(cd, dtype=float).ravel()
    if len(cl) != len(cd):
        raise ValueError("cl and cd must have same length")
    if len(cl) < 3:
        raise ValueError("Need at least 3 data points")

    mask = cd > 0
    cl_f, cd_f = cl[mask], cd[mask]

    cl2 = cl_f**2
    coeffs = np.polyfit(cl2, cd_f, 1)
    K = coeffs[0]
    Cd0 = coeffs[1]

    if K > 0 and Cd0 > 0:
        cl_opt = np.sqrt(Cd0 / K)
        cd_opt = Cd0 + K * cl_opt**2
        ld_max = cl_opt / cd_opt
    else:
        ld_ratio = cl_f / cd_f
        best_idx = np.argmax(ld_ratio)
        cl_opt = cl_f[best_idx]
        cd_opt = cd_f[best_idx]
        ld_max = ld_ratio[best_idx]

    ld_all = cl_f / cd_f

    return DescriptiveResult(
        name="lift_drag_polar",
        value=float(ld_max),
        extra={
            "Cd0": float(Cd0),
            "K": float(K),
            "cl_optimal": float(cl_opt),
            "cd_optimal": float(cd_opt),
            "ld_ratio": ld_all,
            "n_points": len(cl_f),
        },
    )


lifpol = lift_drag_polar


def cheatsheet() -> str:
    return 'lift_drag_polar({}) -> Lift-drag polar.'
