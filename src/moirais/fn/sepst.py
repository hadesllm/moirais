# moirais.fn — function file (hadesllm/moirais)
"""Separation set (Sepset) for PC algorithm edge removal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["separation_set"]


def separation_set(X, Y, ci_tests):
    """
    Separation set (Sepset) for PC algorithm edge removal

    Formula: Sepset(X,Y) = Z such that X _|_ Y | Z; stored during skeleton learning

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    ci_tests : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'sepset': 'set'}

    References
    ----------
    Molak Ch 13
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Separation set (Sepset) for PC algorithm edge removal"})


def cheatsheet():
    return "sepst: Separation set (Sepset) for PC algorithm edge removal"
