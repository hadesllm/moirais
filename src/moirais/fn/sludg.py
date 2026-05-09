"""Rheological flow model (Herschel-Bulkley). 'Sludge no hurry.' -- Sludge"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def herschel_bulkley(
    shear_rate: np.ndarray,
    shear_stress: np.ndarray,
) -> DescriptiveResult:
    """Fit the Herschel-Bulkley rheological model.

    tau = tau_y + K * gamma_dot^n

    where tau_y is yield stress, K is consistency index,
    n is flow behavior index (n<1 shear-thinning, n>1 shear-thickening).

    Parameters
    ----------
    shear_rate : array-like
        Shear rate (gamma_dot) values.
    shear_stress : array-like
        Shear stress (tau) values.

    Returns
    -------
    DescriptiveResult
        With ``value`` = dict(tau_y, K, n) and ``extra``.
    """
    sr = np.asarray(shear_rate, dtype=float).ravel()
    ss = np.asarray(shear_stress, dtype=float).ravel()
    if len(sr) != len(ss):
        raise ValueError("shear_rate and shear_stress must have same length")
    if len(sr) < 4:
        raise ValueError("Need at least 4 data points")

    mask = sr > 0
    sr_f, ss_f = sr[mask], ss[mask]

    def model(gamma_dot, tau_y, K, n):
        return tau_y + K * gamma_dot**n

    try:
        p0 = [ss_f.min() * 0.5, 1.0, 0.5]
        popt, pcov = optimize.curve_fit(
            model,
            sr_f,
            ss_f,
            p0=p0,
            bounds=([0, 0, 0.01], [np.inf, np.inf, 5.0]),
            maxfev=5000,
        )
    except RuntimeError:
        popt = p0
        pcov = np.eye(3) * np.inf

    tau_y, K, n = popt
    fitted = model(sr_f, *popt)
    ss_res = np.sum((ss_f - fitted) ** 2)
    ss_tot = np.sum((ss_f - ss_f.mean()) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-30)

    if n < 1:
        behavior = "shear-thinning"
    elif n > 1:
        behavior = "shear-thickening"
    else:
        behavior = "Bingham"

    return DescriptiveResult(
        name="herschel_bulkley",
        value={"tau_y": float(tau_y), "K": float(K), "n": float(n)},
        extra={"r_squared": float(r_squared), "behavior": behavior, "n_points": len(sr_f), "fitted": fitted},
    )


sludg = herschel_bulkley


def cheatsheet() -> str:
    return "herschel_bulkley({}) -> Rheological flow model (Herschel-Bulkley). 'Sludge no hurry."
