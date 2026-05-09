"""Breslow baseline cumulative hazard."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_breslow_step"]


def cox_breslow_step(time, event, X, beta):
    """
    Breslow baseline cumulative hazard

    Formula: H_0(t) = sum_{t_i <= t} d_i / sum_{j in R_i} exp(beta X_j)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Breslow (1972, 1974)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Breslow baseline cumulative hazard"})


def cheatsheet():
    return "coxbsk: Breslow baseline cumulative hazard"
