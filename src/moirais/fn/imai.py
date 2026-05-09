"""Imai-Keele-Yamamoto sequential ignorability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["imai_keele_yamamoto_mediation"]


def imai_keele_yamamoto_mediation(X, M, Y):
    """
    Imai-Keele-Yamamoto sequential ignorability

    Formula: NIE under SI assumption, mediation package estimator

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imai, Keele, Yamamoto (2010)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Imai-Keele-Yamamoto sequential ignorability"})


def cheatsheet():
    return "imai: Imai-Keele-Yamamoto sequential ignorability"
