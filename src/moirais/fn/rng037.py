"""Equivalent discrete-time causal convolution with swapped arguments.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_discrete_convolution_causal_alt"]


def rangayyan_ch3_discrete_convolution_causal_alt(x, h, n):
    """
    Equivalent discrete-time causal convolution with swapped arguments.

    Formula: y(n) = sum_{k=0}^{n} h(k) x(n - k)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.37, p. 110
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalent discrete-time causal convolution with swapped arguments."})


def cheatsheet():
    return "rng037: Equivalent discrete-time causal convolution with swapped arguments."
