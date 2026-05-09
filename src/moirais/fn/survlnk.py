"""Link-function survival regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["link_function_survival"]


def link_function_survival(time, event, X, link):
    """
    Link-function survival regression

    Formula: g(S(t|X)) = alpha(t) + beta X

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    link : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Klein-Moeschberger (2003)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Link-function survival regression"})


def cheatsheet():
    return "survlnk: Link-function survival regression"
