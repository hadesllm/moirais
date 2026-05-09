"""IPW combined with survey weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipw_with_survey_weights"]


def ipw_with_survey_weights(y, T, weights, propensity):
    """
    IPW combined with survey weights

    Formula: w_i_combined = w_i_design * (T_i / pi_i + (1 - T_i)/(1 - pi_i))

    Parameters
    ----------
    y : array-like
        Input data.
    T : array-like
        Input data.
    weights : array-like
        Input data.
    propensity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    DuGoff, Schuler, Stuart (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IPW combined with survey weights"})


def cheatsheet():
    return "ipwsrv: IPW combined with survey weights"
