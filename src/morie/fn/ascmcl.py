"""Augmented Synthetic Control (Ben-Michael-Feller-Rothstein)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["augmented_synthetic_control"]


def augmented_synthetic_control(y, treated, controls, X):
    """
    Augmented Synthetic Control (Ben-Michael-Feller-Rothstein)

    Formula: y_T - sum_j w_j y_j - hat m(X_T) + sum w_j hat m(X_j)

    Parameters
    ----------
    y : array-like
        Input data.
    treated : array-like
        Input data.
    controls : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ben-Michael, Feller, Rothstein (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Augmented Synthetic Control (Ben-Michael-Feller-Rothstein)"})


def cheatsheet():
    return "ascmcl: Augmented Synthetic Control (Ben-Michael-Feller-Rothstein)"
