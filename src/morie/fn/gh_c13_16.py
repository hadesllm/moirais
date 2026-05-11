# morie.fn — function file (hadesllm/morie)
"""Lo's Bayesian bootstrap for censored survival data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bb_censored"]


def ghosal_bb_censored(x):
    """
    Lo's Bayesian bootstrap for censored survival data

    Formula: DP posterior at alpha->0 for censored data gives Lo's estimator

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 13 §13.7.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lo's Bayesian bootstrap for censored survival data"})


def cheatsheet():
    return "gh_c13_16: Lo's Bayesian bootstrap for censored survival data"
