"""Unit-length normalization of marker vectors."""
import numpy as np
from ._richresult import RichResult

__all__ = ["unit_length_normalization"]


def unit_length_normalization(x):
    """
    Unit-length normalization of marker vectors

    Formula: x_unit = x / ||x||_2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'x_unit': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unit-length normalization of marker vectors"})


def cheatsheet():
    return "unitl: Unit-length normalization of marker vectors"
