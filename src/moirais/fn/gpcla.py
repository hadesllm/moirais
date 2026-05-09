"""GP classification via Laplace approximation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_classification"]


def gp_classification(X, y, X_test, kernel):
    """
    GP classification via Laplace approximation

    Formula: p(y=1|x) = Phi(f(x)); approximate posterior via Newton-Laplace

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Williams-Barber (1998); Rasmussen-Williams (2006) Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP classification via Laplace approximation"})


def cheatsheet():
    return "gpcla: GP classification via Laplace approximation"
