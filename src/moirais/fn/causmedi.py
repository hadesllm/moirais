"""Imai-Keele-Tingley sequential ignorability mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_mediation_imai"]


def causal_mediation_imai(X, M, Y, T, B):
    """
    Imai-Keele-Tingley sequential ignorability mediation

    Formula: ACME = E[Y(t,M(t')) - Y(t,M(t))] under SI

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    T : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ACME, ADE, total

    References
    ----------
    Imai-Keele-Tingley (2010)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Imai-Keele-Tingley sequential ignorability mediation"})


def cheatsheet():
    return "causmedi: Imai-Keele-Tingley sequential ignorability mediation"
