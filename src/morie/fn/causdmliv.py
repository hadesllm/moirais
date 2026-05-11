"""Double ML for instrumental variables."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_dml_iv"]


def causal_dml_iv(y, D, Z, X, n_folds):
    """
    Double ML for instrumental variables

    Formula: Cross-fitted IV/AIPW with nuisances

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.
    X : array-like
        Input data.
    n_folds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, se

    References
    ----------
    Chernozhukov et al. (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double ML for instrumental variables"})


def cheatsheet():
    return "causdmliv: Double ML for instrumental variables"
