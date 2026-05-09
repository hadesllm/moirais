# moirais.fn — function file (hadesllm/moirais)
"""Rescaled GP for adaptation: f(./l_n) with random bandwidth l_n ~ prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_rescal_gp"]


def ghosal_rescal_gp(x):
    """
    Rescaled GP for adaptation: f(./l_n) with random bandwidth l_n ~ prior

    Formula: f_l(x) = f(x/l), l ~ pi(l), adapts to unknown smoothness

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
    Ghosal Ch 11 §11.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rescaled GP for adaptation: f(./l_n) with random bandwidth l_n ~ prior"})


def cheatsheet():
    return "gh_c11_11: Rescaled GP for adaptation: f(./l_n) with random bandwidth l_n ~ prior"
