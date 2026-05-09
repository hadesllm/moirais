# moirais.fn — function file (hadesllm/moirais)
"""Semiparametric Bernstein-von Mises theorem."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bernstein_von_mises"]


def ghosal_bernstein_von_mises(x):
    """
    Semiparametric Bernstein-von Mises theorem

    Formula: sqrt(n)(theta_n - theta_0) -> N(0, I^{-1})

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Ghosal Ch 11
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric Bernstein-von Mises theorem"})


def cheatsheet():
    return "ghbvm: Semiparametric Bernstein-von Mises theorem"
