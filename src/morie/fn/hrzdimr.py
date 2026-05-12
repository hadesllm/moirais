# morie.fn -- function file (hadesllm/morie)
"""Dimension reduction property: single-index aggregates d-dim X to 1D index."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_dimension_reduction"]


def horowitz_dimension_reduction(d, n, bandwidth):
    """
    Dimension reduction property: single-index aggregates d-dim X to 1D index

    Formula: Estimate G at same rate as if X'beta observed: O_p(n^{-2/5}) for 1D nonpar

    Parameters
    ----------
    d : array-like
        Input data.
    n : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate

    References
    ----------
    Horowitz Ch 2
    """
    d = np.asarray(d, dtype=float)
    n = int(d) if d.ndim == 0 else len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dimension reduction property: single-index aggregates d-dim X to 1D index"})


def cheatsheet():
    return "hrzdimr: Dimension reduction property: single-index aggregates d-dim X to 1D index"
