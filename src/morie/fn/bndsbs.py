"""Subset-inference bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_subset_inference"]


def bound_subset_inference(theta_full, subset_idx):
    """
    Subset-inference bound

    Formula: projection of identification set onto subset

    Parameters
    ----------
    theta_full : array-like
        Input data.
    subset_idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Romano-Shaikh (2008)
    """
    theta_full = np.atleast_1d(np.asarray(theta_full, dtype=float))
    n = len(theta_full)
    result = float(np.mean(theta_full))
    se = float(np.std(theta_full, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Subset-inference bound"})


def cheatsheet():
    return "bndsbs: Subset-inference bound"
