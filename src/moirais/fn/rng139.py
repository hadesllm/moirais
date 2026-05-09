"""Wiener filter output expressed as inner product of tap-weight and input vectors.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_output_dot_product"]


def rangayyan_ch3_wiener_output_dot_product(w, x):
    """
    Wiener filter output expressed as inner product of tap-weight and input vectors.

    Formula: d_tilde(n) = w^T * x(n) = x^T(n) * w = <x, w>

    Parameters
    ----------
    w : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.157, p. 174
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener filter output expressed as inner product of tap-weight and input vectors."})


def cheatsheet():
    return "rng139: Wiener filter output expressed as inner product of tap-weight and input vectors."
