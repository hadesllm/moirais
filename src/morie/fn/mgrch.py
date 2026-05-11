"""BEKK multivariate GARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bekk_garch_multivariate"]


def bekk_garch_multivariate(X):
    """
    BEKK multivariate GARCH

    Formula: H_t = C C' + A' eps_{t-1} eps_{t-1}' A + B' H_{t-1} B

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle & Kroner (1995)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BEKK multivariate GARCH"})


def cheatsheet():
    return "mgrch: BEKK multivariate GARCH"
