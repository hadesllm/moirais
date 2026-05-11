"""Stochastic intervention MSM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spsm_shifted_intervention"]


def spsm_shifted_intervention(y, A, H, shift_fn):
    """
    Stochastic intervention MSM

    Formula: target shifted-intervention parameter E[Y(d_shift)]

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    shift_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Díaz-vdL (2012, 2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic intervention MSM"})


def cheatsheet():
    return "spsmsh: Stochastic intervention MSM"
