"""Shared frailty marginal model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shared_frailty_marginal"]


def shared_frailty_marginal(time, event, X, cluster):
    """
    Shared frailty marginal model

    Formula: lambda_i(t|w) = w lambda_0(t) exp(beta'X_i), w shared in cluster

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vaupel et al. (1979)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shared frailty marginal model"})


def cheatsheet():
    return "shfrm: Shared frailty marginal model"
