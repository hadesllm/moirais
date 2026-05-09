# moirais.fn — function file (hadesllm/moirais)
"""Partial Kendall's tau."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def partial_kendall(
    x,
    y,
    z,
) -> ESRes:
    """Partial Kendall's tau controlling for z.

    tau_xy.z = (tau_xy - tau_xz * tau_yz) /
               sqrt((1 - tau_xz^2)(1 - tau_yz^2))

    Parameters
    ----------
    x, y : array-like
        Primary variables.
    z : array-like
        Control variable.

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations.")

    tau_xy, _ = sp_stats.kendalltau(x, y)
    tau_xz, _ = sp_stats.kendalltau(x, z)
    tau_yz, _ = sp_stats.kendalltau(y, z)

    denom = np.sqrt((1 - tau_xz**2) * (1 - tau_yz**2))
    if denom < 1e-12:
        partial_tau = 0.0
    else:
        partial_tau = (tau_xy - tau_xz * tau_yz) / denom

    return ESRes(
        measure="partial_kendall",
        estimate=float(partial_tau),
        n=n,
        extra={
            "tau_xy": float(tau_xy),
            "tau_xz": float(tau_xz),
            "tau_yz": float(tau_yz),
        },
    )


pknds = partial_kendall


def cheatsheet() -> str:
    return "partial_kendall(x, y, z) -> Partial Kendall's tau."
