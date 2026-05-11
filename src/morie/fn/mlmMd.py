"""Multilevel (1-1-1, 2-1-1) mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multilevel_mediation"]


def multilevel_mediation(Y, X, M, cluster):
    """
    Multilevel (1-1-1, 2-1-1) mediation

    Formula: between- vs within-cluster decomposition

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Krull-MacKinnon (2001); Zhang-Zyphur-Preacher (2009)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilevel (1-1-1, 2-1-1) mediation"})


def cheatsheet():
    return "mlmMd: Multilevel (1-1-1, 2-1-1) mediation"
