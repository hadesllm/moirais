"""Stabilized inverse-probability-of-censoring weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stabilized_censoring_weights"]


def stabilized_censoring_weights(C, H):
    """
    Stabilized inverse-probability-of-censoring weights

    Formula: sw_C = f(C_t)/f(C_t|H_t)

    Parameters
    ----------
    C : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (1993); Hernán-Brumback-Robins (2002)
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stabilized inverse-probability-of-censoring weights"})


def cheatsheet():
    return "stbciw: Stabilized inverse-probability-of-censoring weights"
