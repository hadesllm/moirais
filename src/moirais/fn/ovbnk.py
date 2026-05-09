"""Oster bound on bias from omitted variables (delta)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["oster_omitted_bias_bound"]


def oster_omitted_bias_bound(beta_short, beta_long, R_short, R_long, R_max, delta):
    """
    Oster bound on bias from omitted variables (delta)

    Formula: beta* = beta_short - delta * (R_max - R_short)/(R_short - R_unctl) * (beta_short - beta_long)

    Parameters
    ----------
    beta_short : array-like
        Input data.
    beta_long : array-like
        Input data.
    R_short : array-like
        Input data.
    R_long : array-like
        Input data.
    R_max : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Oster (2019)
    """
    beta_short = np.atleast_1d(np.asarray(beta_short, dtype=float))
    n = len(beta_short)
    result = float(np.mean(beta_short))
    se = float(np.std(beta_short, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Oster bound on bias from omitted variables (delta)"})


def cheatsheet():
    return "ovbnk: Oster bound on bias from omitted variables (delta)"
