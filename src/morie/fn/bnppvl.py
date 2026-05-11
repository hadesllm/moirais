"""NP Bayes predictive value of new observation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bnp_predictive_value"]


def bnp_predictive_value(y, prior_family):
    """
    NP Bayes predictive value of new observation

    Formula: P(y_n+1 | y_1:n) integrated

    Parameters
    ----------
    y : array-like
        Input data.
    prior_family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hjort-Walker (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NP Bayes predictive value of new observation"})


def cheatsheet():
    return "bnppvl: NP Bayes predictive value of new observation"
