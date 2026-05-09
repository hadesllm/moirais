"""ATT inverse probability of treatment weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_iptw_attweights"]


def causal_iptw_attweights(treat, ps):
    """
    ATT inverse probability of treatment weights

    Formula: w_i = e(x_i)/(1-e(x_i)) for control, 1 for treated

    Parameters
    ----------
    treat : array-like
        Input data.
    ps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, ess

    References
    ----------
    Hernán-Robins (2020)
    """
    treat = np.atleast_1d(np.asarray(treat, dtype=float))
    n = len(treat)
    result = float(np.mean(treat))
    se = float(np.std(treat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ATT inverse probability of treatment weights"})


def cheatsheet():
    return "causipsw: ATT inverse probability of treatment weights"
