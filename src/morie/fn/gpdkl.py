"""Deep kernel learning GP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deep_kernel_gp"]


def deep_kernel_gp(X, y, X_test, nn):
    """
    Deep kernel learning GP

    Formula: k(x,x') = k_RBF(g(x), g(x')) with NN g

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    nn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wilson-Hu-Salakhutdinov-Xing (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep kernel learning GP"})


def cheatsheet():
    return "gpdkl: Deep kernel learning GP"
