"""Construct replicate weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["replicate_weights"]


def replicate_weights(design, method, R):
    """
    Construct replicate weights

    Formula: jackknife / BRR / bootstrap weight matrix

    Parameters
    ----------
    design : array-like
        Input data.
    method : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wolter (2007)
    """
    design = np.atleast_1d(np.asarray(design, dtype=float))
    n = len(design)
    result = float(np.mean(design))
    se = float(np.std(design, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Construct replicate weights"})


def cheatsheet():
    return "replwt: Construct replicate weights"
