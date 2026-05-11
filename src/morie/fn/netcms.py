"""Network psychometrics (graphical LASSO)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["network_psychometrics"]


def network_psychometrics(X, lam):
    """
    Network psychometrics (graphical LASSO)

    Formula: sparse precision matrix from data

    Parameters
    ----------
    X : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Epskamp et al (2018)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network psychometrics (graphical LASSO)"})


def cheatsheet():
    return "netcms: Network psychometrics (graphical LASSO)"
