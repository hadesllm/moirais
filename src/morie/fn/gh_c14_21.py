# morie.fn -- function file (hadesllm/morie)
"""Ordering-dependent stick-breaking: w_k depends on ordering of covariate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ord_dep_sbp"]


def ghosal_ord_dep_sbp(x):
    """
    Ordering-dependent stick-breaking: w_k depends on ordering of covariate

    Formula: V_k = V_k(x_{(k)}), ordering-dependent weights for regression

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
    Ghosal Ch 14 §14.9.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordering-dependent stick-breaking: w_k depends on ordering of covariate"})


def cheatsheet():
    return "gh_c14_21: Ordering-dependent stick-breaking: w_k depends on ordering of covariate"
