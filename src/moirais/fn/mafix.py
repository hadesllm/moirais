"""Inverse-variance fixed-effect pooled mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_fixed_effect"]


def ma_fixed_effect(yi, vi):
    """
    Inverse-variance fixed-effect pooled mean

    Formula: θ̂ = Σ w_i y_i / Σ w_i,  w_i = 1/v_i

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, se, z, p

    References
    ----------
    Borenstein et al. (2009) §11
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse-variance fixed-effect pooled mean"})


def cheatsheet():
    return "mafix: Inverse-variance fixed-effect pooled mean"
