"""Inverse logit back to proportion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_logit_inverse"]


def ma_logit_inverse(z):
    """
    Inverse logit back to proportion

    Formula: p = exp(z)/(1+exp(z))

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Nyaga et al. (2014)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse logit back to proportion"})


def cheatsheet():
    return "matransi: Inverse logit back to proportion"
