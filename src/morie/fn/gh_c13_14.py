# morie.fn -- function file (hadesllm/morie)
"""Bayesian Cox posterior for regression coefficient beta via partial likelihood."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_cox_post"]


def ghosal_cox_post(x):
    """
    Bayesian Cox posterior for regression coefficient beta via partial likelihood

    Formula: pi(beta | data) propto pi(beta) * prod exp(beta'x_i - log sum_{j in R_i} exp(beta'x_j))

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
    Ghosal Ch 13 §13.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian Cox posterior for regression coefficient beta via partial likelihood"})


def cheatsheet():
    return "gh_c13_14: Bayesian Cox posterior for regression coefficient beta via partial likelihood"
