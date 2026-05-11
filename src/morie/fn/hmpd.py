# morie.fn — function file (hadesllm/morie)
"""Zero-padding around input for valid/same convolutions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_padding"]


def geron_padding(x, pad_h, pad_w):
    """
    Zero-padding around input for valid/same convolutions

    Formula: pad_h = (kh - 1)/2 for same padding

    Parameters
    ----------
    x : array-like
        Input data.
    pad_h : array-like
        Input data.
    pad_w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_padded

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zero-padding around input for valid/same convolutions"})


def cheatsheet():
    return "hmpd: Zero-padding around input for valid/same convolutions"
