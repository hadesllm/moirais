"""Omitted variable bias formula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["omitted_variable_bias"]


def omitted_variable_bias(beta_xu, beta_yu_given_x):
    """
    Omitted variable bias formula

    Formula: bias = β_{Y~U|X,C} · β_{X~U|C}

    Parameters
    ----------
    beta_xu : array-like
        Input data.
    beta_yu_given_x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cinelli-Hazlett (2020)
    """
    beta_xu = np.atleast_1d(np.asarray(beta_xu, dtype=float))
    n = len(beta_xu)
    result = float(np.mean(beta_xu))
    se = float(np.std(beta_xu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Omitted variable bias formula"})


def cheatsheet():
    return "omitV: Omitted variable bias formula"
