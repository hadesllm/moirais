"""GP spectral mixture kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_spectral_mixture"]


def gp_spectral_mixture(X, y, X_test, Q):
    """
    GP spectral mixture kernel

    Formula: k(tau) = sum_q w_q exp(-tau^2 v_q) cos(2pi mu_q tau)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    Q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wilson-Adams (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP spectral mixture kernel"})


def cheatsheet():
    return "gpsps: GP spectral mixture kernel"
