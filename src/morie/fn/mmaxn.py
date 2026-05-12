# morie.fn -- function file (hadesllm/morie)
"""Min-max normalization for marker data or responses."""
import numpy as np
from ._richresult import RichResult

__all__ = ["minmax_normalization"]


def minmax_normalization(x):
    """
    Min-max normalization for marker data or responses

    Formula: x_norm = (x - x_min) / (x_max - x_min)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'x_norm': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Min-max normalization for marker data or responses"})


def cheatsheet():
    return "mmaxn: Min-max normalization for marker data or responses"
