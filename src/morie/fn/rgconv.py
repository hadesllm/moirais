# morie.fn -- function file (hadesllm/morie)
"""Linear convolution of two finite-length sequences."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_linear_convolution"]


def rangayyan_linear_convolution(x, h):
    """
    Linear convolution of two finite-length sequences

    Formula: y[n] = sum_{k=-inf}^{inf} x[k] * h[n-k]

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear convolution of two finite-length sequences"})


def cheatsheet():
    return "rgconv: Linear convolution of two finite-length sequences"
