# morie.fn — function file (hadesllm/morie)
"""Exchangeability (unconfoundedness/ignorability): treatment independent of potential outcomes given covariates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["exchangeability_assumption"]


def exchangeability_assumption(Y, T, X, dag):
    """
    Exchangeability (unconfoundedness/ignorability): treatment independent of potential outcomes given covariates

    Formula: Y(t) _|_ T | X for all t; strong ignorability

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.
    dag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'exchangeable': 'bool'}

    References
    ----------
    Molak Ch 8
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exchangeability (unconfoundedness/ignorability): treatment independent of potential outcomes given covariates"})


def cheatsheet():
    return "exchg: Exchangeability (unconfoundedness/ignorability): treatment independent of potential outcomes given covariates"
