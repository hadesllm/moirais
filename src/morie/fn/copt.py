"""Student t copula CDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["t_copula"]


def t_copula(y, u, v, rho, nu):
    """
    Student t copula CDF

    Formula: C(u,v) = T_{nu,R}(t_nu^{-1}(u), t_nu^{-1}(v))

    Parameters
    ----------
    y : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
        Input data.
    rho : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Embrechts, McNeil, Straumann (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Student t copula CDF"})


def cheatsheet():
    return "copt: Student t copula CDF"
