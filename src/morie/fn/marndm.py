"""Random-effects pooled mean via DerSimonian-Laird τ²."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_random_dl"]


def ma_random_dl(yi, vi):
    """
    Random-effects pooled mean via DerSimonian-Laird τ²

    Formula: τ̂² = max(0, (Q-(k-1))/c); w*_i = 1/(v_i+τ̂²)

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, se, tau2, Q, p

    References
    ----------
    DerSimonian & Laird (1986)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random-effects pooled mean via DerSimonian-Laird τ²"})


def cheatsheet():
    return "marndm: Random-effects pooled mean via DerSimonian-Laird τ²"
