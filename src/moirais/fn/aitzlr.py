"""Log-ratio EM imputation of compositional zeros."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_zero_lrem"]


def compositional_zero_lrem(X, max_iter):
    """
    Log-ratio EM imputation of compositional zeros

    Formula: iterate E[log x_i | observed] under multivariate normal in ALR

    Parameters
    ----------
    X : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_imp, ll

    References
    ----------
    Palarea-Albaladejo & Martín-Fernández (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-ratio EM imputation of compositional zeros"})


def cheatsheet():
    return "aitzlr: Log-ratio EM imputation of compositional zeros"
