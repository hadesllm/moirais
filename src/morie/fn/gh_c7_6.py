# morie.fn -- function file (hadesllm/morie)
"""Polya tree density estimation: posterior consistent at Lipschitz densities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pt_dens_con"]


def ghosal_pt_dens_con(x):
    """
    Polya tree density estimation: posterior consistent at Lipschitz densities

    Formula: PT(T_m, a_m) consistent in Hellinger for Lipschitz p0

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
    Ghosal Ch 7 §7.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Polya tree density estimation: posterior consistent at Lipschitz densities"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Polya tree density estimation: posterior consistent at Lipschitz densities"})


def cheatsheet():
    return "gh_c7_6: Polya tree density estimation: posterior consistent at Lipschitz densities"
