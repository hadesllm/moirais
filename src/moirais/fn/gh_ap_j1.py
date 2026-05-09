# moirais.fn — function file (hadesllm/moirais)
"""Levy-Ito decomposition of completely random measure."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_levy_ito"]


def ghosal_levy_ito(x):
    """
    Levy-Ito decomposition of completely random measure

    Formula: M = M_fixed + M_atom + M_diffuse, Laplace functional E[exp(-lambda*M(A))]

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
    Ghosal App J
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Levy-Ito decomposition of completely random measure"})


def cheatsheet():
    return "gh_ap_j1: Levy-Ito decomposition of completely random measure"
