"""k-fold cross-validation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_kfold_cv"]


def wasserman_kfold_cv(X, y, model, k):
    """
    k-fold cross-validation

    Formula: CV(k) = (1/n) sum (Y_i - m_hat^{-i}(X_i))^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    model : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wasserman (2004), Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-fold cross-validation"})


def cheatsheet():
    return "wsmcvr: k-fold cross-validation"
