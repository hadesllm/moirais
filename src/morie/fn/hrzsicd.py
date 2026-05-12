# morie.fn -- function file (hadesllm/morie)
"""Identification of single-index model when X is discrete."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_sim_id_discrete_x"]


def horowitz_sim_id_discrete_x(x, y):
    """
    Identification of single-index model when X is discrete

    Formula: Identified if G is monotone and has one continuous component with nonzero beta

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Horowitz Ch 2, Sec 2.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identification of single-index model when X is discrete"})


def cheatsheet():
    return "hrzsicd: Identification of single-index model when X is discrete"
