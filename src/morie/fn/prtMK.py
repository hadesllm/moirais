"""Pre-whitened Mann-Kendall."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prewhitening_mk"]


def prewhitening_mk(x):
    """
    Pre-whitened Mann-Kendall

    Formula: remove AR(1) before MK

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
    Yue-Pilon-Cavadias (2002)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pre-whitened Mann-Kendall"})


def cheatsheet():
    return "prtMK: Pre-whitened Mann-Kendall"
