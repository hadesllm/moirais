# morie.fn — function file (hadesllm/morie)
"""Empirical Bayes nonparametric."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_empirical_bayes"]


def ghosal_empirical_bayes(x):
    """
    Empirical Bayes nonparametric

    Formula: alpha_hat = argmax marginal likelihood

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
    Ghosal Ch 15
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical Bayes nonparametric"})


def cheatsheet():
    return "ghebp: Empirical Bayes nonparametric"
