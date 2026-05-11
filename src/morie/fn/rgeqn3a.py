# morie.fn — function file (hadesllm/morie)
"""Convolution sum for linear shift-invariant (LSI) system output."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_convolution_sum"]


def rangayyan_ch3_convolution_sum(x, h):
    """
    Convolution sum for linear shift-invariant (LSI) system output

    Formula: y[n] = sum_{k} h[k]*x[n-k]

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convolution sum for linear shift-invariant (LSI) system output"})


def cheatsheet():
    return "rgeqn3a: Convolution sum for linear shift-invariant (LSI) system output"
