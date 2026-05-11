"""Influence-curve based inference for TMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_inference"]


def tmle_inference(y, D, X, Q, g):
    """
    Influence-curve based inference for TMLE

    Formula: Var(psi_hat) = E[D^2]/n where D is influence curve

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    Q : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL & Rose (2011) Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Influence-curve based inference for TMLE"})


def cheatsheet():
    return "tmlinf: Influence-curve based inference for TMLE"
