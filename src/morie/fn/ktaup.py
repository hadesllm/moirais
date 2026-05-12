# morie.fn — function file (hadesllm/morie)
"""Kendall partial-tau correlation (Gibbons Ch 12.6).

Partial Kendall tau between X and Y controlling for Z:

    tau_xy.z = (tau_xy - tau_xz * tau_yz) /
               sqrt((1 - tau_xz^2) * (1 - tau_yz^2))

Significance is tested via the standard Kendall-tau normal-
approximation z-statistic on the partial tau (Gibbons, eq. 12.6.2).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["kendall_tau_partial"]


def kendall_tau_partial(x, y, z):
    """Partial Kendall tau of X, Y given Z.

    Parameters
    ----------
    x, y, z : array-like
        Three numeric vectors of equal length.

    Returns
    -------
    RichResult with payload:
        statistic : partial tau (tau_xy.z)
        p_value   : two-sided p-value (normal approximation)
        tau_xy, tau_xz, tau_yz : marginal tau values
        n         : sample size
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = min(len(x), len(y), len(z))
    if n < 4:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan,
            "tau_xy": np.nan, "tau_xz": np.nan, "tau_yz": np.nan,
            "n": n, "method": "Kendall partial tau",
        })
    x, y, z = x[:n], y[:n], z[:n]
    tau_xy = stats.kendalltau(x, y).statistic
    tau_xz = stats.kendalltau(x, z).statistic
    tau_yz = stats.kendalltau(y, z).statistic
    denom = np.sqrt((1.0 - tau_xz ** 2) * (1.0 - tau_yz ** 2))
    if denom == 0 or not np.isfinite(denom):
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan,
            "tau_xy": float(tau_xy), "tau_xz": float(tau_xz),
            "tau_yz": float(tau_yz), "n": n,
            "method": "Kendall partial tau",
        })
    tau_p = (tau_xy - tau_xz * tau_yz) / denom
    # Use the Kendall-tau normal approx (z = tau * sqrt(9n(n-1) / (2(2n+5))))
    z_stat = tau_p * np.sqrt(9.0 * n * (n - 1) / (2.0 * (2 * n + 5)))
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return RichResult(payload={
        "statistic": float(tau_p),
        "p_value": float(p),
        "tau_xy": float(tau_xy),
        "tau_xz": float(tau_xz),
        "tau_yz": float(tau_yz),
        "z": float(z_stat),
        "n": n,
        "method": "Kendall partial tau (xy controlling z)",
    })


def cheatsheet():
    return "ktaup: Kendall partial tau (xy.z)"


# CANONICAL TEST
# >>> kendall_tau_partial([1,2,3,4,5], [1,2,3,4,5], [5,4,3,2,1])
# tau_xy=1, tau_xz=tau_yz=-1, denom=0 -> partial undefined (returns nan).
# Use real example with z mildly related to keep denom nonzero.
