# morie.fn -- function file (hadesllm/morie)
"""Parameters."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ._containers import DescriptiveResult


def logistic_growth(
    t: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    p0: tuple[float, float, float] | None = None,
) -> DescriptiveResult:
    """Fit a logistic growth curve :math:`y(t) = K / (1 + e^{-r(t - t_0)})`.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values (population, biomass, etc.).
    p0 : tuple or None
        Initial guesses (K, r, t0).  Auto-estimated if None.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``K`` (carrying capacity), ``r`` (growth rate),
        ``t0`` (inflection time), ``fitted``, ``residuals``, ``r_squared``.
    """
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(t_arr) != len(y_arr) or len(t_arr) < 3:
        raise ValueError("t and y must have same length >= 3")

    def _logistic(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))

    if p0 is None:
        K0 = float(y_arr.max()) * 1.1
        t0_0 = float(t_arr[len(t_arr) // 2])
        r0 = 1.0
        p0 = (K0, r0, t0_0)

    try:
        popt, pcov = curve_fit(_logistic, t_arr, y_arr, p0=p0, maxfev=5000)
    except RuntimeError as e:
        raise ValueError(f"Curve fitting failed: {e}") from e

    K, r, t0_fit = popt
    fitted = _logistic(t_arr, K, r, t0_fit)
    resid = y_arr - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DescriptiveResult(
        name="logistic_growth",
        value={
            "K": float(K),
            "r": float(r),
            "t0": float(t0_fit),
            "fitted": fitted,
            "residuals": resid,
            "r_squared": r2,
        },
        extra={"n": len(t_arr), "se": np.sqrt(np.diag(pcov)).tolist()},
    )


loggro = logistic_growth


def cheatsheet() -> str:
    return 'logistic_growth({}) -> Logistic growth model.'
