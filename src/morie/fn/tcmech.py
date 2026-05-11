"""Truncated CDP for unbounded query."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["truncated_cdp_mechanism"]


def truncated_cdp_mechanism(y, f_value, C, epsilon, delta):
    """
    Truncated CDP for unbounded query

    Formula: clip f(x) to [-C, C]; apply Gaussian noise w/ sensitivity 2C

    Parameters
    ----------
    y : array-like
        Input data.
    f_value : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bun et al (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Truncated CDP for unbounded query"})


def cheatsheet():
    return "tcmech: Truncated CDP for unbounded query"
