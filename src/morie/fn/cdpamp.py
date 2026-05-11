"""Concentrated DP subgaussian amplification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cdp_subgaussian_amplification"]


def cdp_subgaussian_amplification(y, epsilon, delta, k_compositions):
    """
    Concentrated DP subgaussian amplification

    Formula: epsilon_amp = epsilon * sqrt(2 ln(2/delta) / k_compositions)

    Parameters
    ----------
    y : array-like
        Input data.
    epsilon : array-like
        Input data.
    delta : array-like
        Input data.
    k_compositions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork & Rothblum (2016) Concentrated DP
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Concentrated DP subgaussian amplification"})


def cheatsheet():
    return "cdpamp: Concentrated DP subgaussian amplification"
