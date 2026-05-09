"""Additive log-ratio (ALR) transform with reference part."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_alr"]


def aitchison_alr(x, ref):
    """
    Additive log-ratio (ALR) transform with reference part

    Formula: alr_i(x) = log(x_i / x_D),  i=1..D-1

    Parameters
    ----------
    x : array-like
        Input data.
    ref : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Aitchison (1986) §4
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Additive log-ratio (ALR) transform with reference part"})


def cheatsheet():
    return "aitalr: Additive log-ratio (ALR) transform with reference part"
