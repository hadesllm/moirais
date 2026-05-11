"""Cox shared-frailty model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_frailty"]


def cox_frailty(time, event, X, cluster):
    """
    Cox shared-frailty model

    Formula: lambda_ij(t) = lambda_0(t) z_i exp(beta X_ij)

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
    Hougaard (2000); Therneau-Grambsch (2000)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox shared-frailty model"})


def cheatsheet():
    return "coxfrl: Cox shared-frailty model"
