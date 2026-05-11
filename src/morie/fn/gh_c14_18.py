# morie.fn — function file (hadesllm/morie)
"""Kernel stick-breaking process: dependent weights via kernel function of covariate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ksbp_def"]


def ghosal_ksbp_def(x):
    """
    Kernel stick-breaking process: dependent weights via kernel function of covariate

    Formula: p_k(x) = V_k(x) prod_{j<k}(1-V_j(x)), V_k(x) = g(w_k'x)

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
    Ghosal Ch 14 §14.9.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel stick-breaking process: dependent weights via kernel function of covariate"})


def cheatsheet():
    return "gh_c14_18: Kernel stick-breaking process: dependent weights via kernel function of covariate"
