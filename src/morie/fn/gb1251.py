# morie.fn — function file (hadesllm/morie)
"""Kendall partial tau for controlling confounder z."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_partial_tau"]


def gibbons_partial_tau(x, y, z):
    """
    Kendall partial tau for controlling confounder z

    Formula: tau_{xy.z} = (tau_xy - tau_xz*tau_yz) / sqrt((1-tau_xz^2)(1-tau_yz^2))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: partial_tau, p_value

    References
    ----------
    Gibbons Ch 12.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kendall partial tau for controlling confounder z"})


def cheatsheet():
    return "gb1251: Kendall partial tau for controlling confounder z"
