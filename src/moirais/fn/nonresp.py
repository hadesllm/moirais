"""Nonresponse adjustment via response propensity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nonresponse_adjustment"]


def nonresponse_adjustment(y, weights, propensity):
    """
    Nonresponse adjustment via response propensity

    Formula: w_i_adj = w_i / hat phi_i; phi = P(respond | X)

    Parameters
    ----------
    y : array-like
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
    Little & Rubin (2002) §15
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonresponse adjustment via response propensity"})


def cheatsheet():
    return "nonresp: Nonresponse adjustment via response propensity"
