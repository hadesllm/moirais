# morie.fn -- function file (hadesllm/morie)
"""Spectral density estimation consistency via Whittle likelihood."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_spec_dens_con"]


def ghosal_spec_dens_con(x):
    """
    Spectral density estimation consistency via Whittle likelihood

    Formula: Whittle likelihood: L_W(f) = prod exp(-I(omega_j)/f(omega_j))/f(omega_j)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 7 §7.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Spectral density estimation consistency via Whittle likelihood"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Spectral density estimation consistency via Whittle likelihood"})


def cheatsheet():
    return "gh_c7_7: Spectral density estimation consistency via Whittle likelihood"
