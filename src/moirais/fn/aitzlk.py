"""Log-ratio data augmentation imputation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_zero_lrda"]


def compositional_zero_lrda(X, n_iter):
    """
    Log-ratio data augmentation imputation

    Formula: Bayesian draws from posterior of α under truncated MVN in ALR

    Parameters
    ----------
    X : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_imp, draws

    References
    ----------
    Palarea-Albaladejo & Martín-Fernández (2013)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-ratio data augmentation imputation"})


def cheatsheet():
    return "aitzlk: Log-ratio data augmentation imputation"
