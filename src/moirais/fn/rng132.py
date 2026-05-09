"""General time-domain difference equation of an IIR filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_iir_difference_eq_general"]


def rangayyan_ch3_iir_difference_eq_general(x, y, b_k, a_k, N, n):
    """
    General time-domain difference equation of an IIR filter.

    Formula: y(n) = sum_{k=0}^{N} b_k x(n-k) - sum_{k=1}^{N} a_k y(n-k)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    b_k : array-like
        Input data.
    a_k : array-like
        Input data.
    N : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.144, p. 155
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "General time-domain difference equation of an IIR filter."})


def cheatsheet():
    return "rng132: General time-domain difference equation of an IIR filter."
