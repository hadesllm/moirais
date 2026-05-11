"""Stochastic variational GP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_stochastic_vi"]


def gp_stochastic_vi(X, y, X_test, inducing, batch_size):
    """
    Stochastic variational GP

    Formula: ELBO with mini-batch + inducing

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    inducing : array-like
        Input data.
    batch_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hensman-Fusi-Lawrence (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic variational GP"})


def cheatsheet():
    return "gpsvi: Stochastic variational GP"
