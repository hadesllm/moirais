"""DP Bayesian release of posterior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_bayesian_mechanism"]


def dp_bayesian_mechanism(y, posterior_sample, epsilon):
    """
    DP Bayesian release of posterior

    Formula: sample theta ~ posterior; release with appropriate noise

    Parameters
    ----------
    y : array-like
        Input data.
    posterior_sample : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang, Fienberg, Smola (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP Bayesian release of posterior"})


def cheatsheet():
    return "bayesm: DP Bayesian release of posterior"
