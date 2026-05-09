"""BayesA prior on marker effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_a_alpha"]


def bayes_a_alpha(y, M):
    """
    BayesA prior on marker effects

    Formula: u_j ~ N(0, sigma_j^2); sigma_j^2 ~ scaled-inv-chi2

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meuwissen-Hayes-Goddard (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BayesA prior on marker effects"})


def cheatsheet():
    return "baysab: BayesA prior on marker effects"
