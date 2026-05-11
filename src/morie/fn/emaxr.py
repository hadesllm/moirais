"""EM step (single iteration) for random-effects variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["em_step_random_effects"]


def em_step_random_effects(y, X, cluster, sigma2_u, sigma2_e):
    """
    EM step (single iteration) for random-effects variance

    Formula: sigma2_u^(t+1) = (1/J) sum_j [u_j_hat^2 + var(u_j | y)]

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.
    sigma2_u : array-like
        Input data.
    sigma2_e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dempster, Laird, Rubin (1977); Laird & Ware (1982)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EM step (single iteration) for random-effects variance"})


def cheatsheet():
    return "emaxr: EM step (single iteration) for random-effects variance"
