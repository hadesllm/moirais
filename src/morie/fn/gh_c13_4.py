# morie.fn -- function file (hadesllm/morie)
"""Discrete-time Beta process: product of independent Beta random variables."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bp_discrete"]


def ghosal_bp_discrete(x):
    """
    Discrete-time Beta process: product of independent Beta random variables

    Formula: H(t) = sum_{s<=t} dH(s), dH(s_k) ~ Be(c_k*h_k, c_k*(1-h_k)) independent

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
    Ghosal Ch 13 §13.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete-time Beta process: product of independent Beta random variables"})


def cheatsheet():
    return "gh_c13_4: Discrete-time Beta process: product of independent Beta random variables"
