# morie.fn -- function file (hadesllm/morie)
"""Poisson-Kingman process: discrete random measure via subordinator construction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pk_process"]


def ghosal_pk_process(x):
    """
    Poisson-Kingman process: discrete random measure via subordinator construction

    Formula: G = sum_k (J_k/T) delta_{theta_k}, J_k from Poisson process with Levy measure rho

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
    Ghosal Ch 14 §14.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson-Kingman process: discrete random measure via subordinator construction"})


def cheatsheet():
    return "gh_c14_12: Poisson-Kingman process: discrete random measure via subordinator construction"
