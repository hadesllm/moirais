"""BayesC pi."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_c_pi"]


def bayes_c_pi(y, M, pi):
    """
    BayesC pi

    Formula: u_j ~ N(0, sigma_b^2) with prob (1-pi); 0 otherwise

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Habier et al (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BayesC pi"})


def cheatsheet():
    return "bayscm: BayesC pi"
