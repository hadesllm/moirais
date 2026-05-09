"""Method-of-moments fit of Dirichlet α."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_fit_mom"]


def dirichlet_fit_mom(X):
    """
    Method-of-moments fit of Dirichlet α

    Formula: α̂_i = m_i s, s = (m_1(1-m_1)/v_1) - 1

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha

    References
    ----------
    Minka (2000)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Method-of-moments fit of Dirichlet α"})


def cheatsheet():
    return "aitdrf: Method-of-moments fit of Dirichlet α"
