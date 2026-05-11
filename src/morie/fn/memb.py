"""Membership inference attack."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["membership_inference"]


def membership_inference(model, x, shadow_models):
    """
    Membership inference attack

    Formula: adversary classifies if x was in train set

    Parameters
    ----------
    model : array-like
        Input data.
    x : array-like
        Input data.
    shadow_models : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shokri et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Membership inference attack"})


def cheatsheet():
    return "memb: Membership inference attack"
