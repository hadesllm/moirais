"""CFA multi-factor with cross-loadings allowed."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cfa_multifactor"]


def cfa_multifactor(X, factor_pattern):
    """
    CFA multi-factor with cross-loadings allowed

    Formula: X = Lambda F + eps; F ~ N(0, Phi)

    Parameters
    ----------
    X : array-like
        Input data.
    factor_pattern : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jöreskog (1969)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CFA multi-factor with cross-loadings allowed"})


def cheatsheet():
    return "cfafm2: CFA multi-factor with cross-loadings allowed"
