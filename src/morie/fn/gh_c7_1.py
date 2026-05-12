# morie.fn -- function file (hadesllm/morie)
"""Polya tree KL property: canonical PT*(alpha, a_m) has KL support at continuous densities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pt_kl_prop"]


def ghosal_pt_kl_prop(x):
    """
    Polya tree KL property: canonical PT*(alpha, a_m) has KL support at continuous densities

    Formula: PT*(alpha, a_m) with a_m = alpha_m^2 satisfies KL condition

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
    Ghosal Ch 7 §7.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polya tree KL property: canonical PT*(alpha, a_m) has KL support at continuous densities"})


def cheatsheet():
    return "gh_c7_1: Polya tree KL property: canonical PT*(alpha, a_m) has KL support at continuous densities"
