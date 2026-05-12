# morie.fn -- function file (hadesllm/morie)
"""Empirical distribution function (EDF)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_ecdf"]


def fauzi_ecdf(x):
    """
    Empirical distribution function (EDF)

    Formula: F_n(x) = (1/n) sum I(X_i <= x)

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
    Fauzi Ch 2, Eq 2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical distribution function (EDF)"})


def cheatsheet():
    return "fzecdf: Empirical distribution function (EDF)"
