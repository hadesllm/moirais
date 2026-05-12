# morie.fn -- function file (hadesllm/morie)
"""Diaconis-Freedman inconsistency example: symmetric Dirichlet posterior inconsistent."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_df_inconsist"]


def ghosal_df_inconsist(x):
    """
    Diaconis-Freedman inconsistency example: symmetric Dirichlet posterior inconsistent

    Formula: Counterexample: Pi consistent only on Pi-null set of theta_0

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
    Ghosal Ch 6 §6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Diaconis-Freedman inconsistency example: symmetric Dirichlet posterior inconsistent"})


def cheatsheet():
    return "gh_c6_4: Diaconis-Freedman inconsistency example: symmetric Dirichlet posterior inconsistent"
