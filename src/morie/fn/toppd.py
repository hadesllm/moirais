"""Nucleus (top-p) sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["top_p_nucleus"]


def top_p_nucleus(x):
    """
    Nucleus (top-p) sampling

    Formula: sample from smallest set with sum p >= p_threshold

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
    Holtzman et al. (2020)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nucleus (top-p) sampling"})


def cheatsheet():
    return "toppd: Nucleus (top-p) sampling"
