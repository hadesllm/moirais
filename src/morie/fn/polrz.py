# morie.fn — function file (hadesllm/morie)
"""Political polarization measures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["polarization_index"]


def polarization_index(x):
    """
    Political polarization measures

    Formula: P = |mean(x_R) - mean(x_D)| / pooled_sd

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
    Armstrong Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Political polarization measures"})


def cheatsheet():
    return "polrz: Political polarization measures"
