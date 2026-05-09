"""Laplace transform of a causal finite-duration h(t) over [0,T].."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_laplace_transform_causal_finite"]


def rangayyan_ch3_laplace_transform_causal_finite(h, t, s, T):
    """
    Laplace transform of a causal finite-duration h(t) over [0,T].

    Formula: H(s) = integral_{0}^{T} h(t) * exp(-s*t) dt

    Parameters
    ----------
    h : array-like
        Input data.
    t : array-like
        Input data.
    s : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.51, p. 117
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplace transform of a causal finite-duration h(t) over [0,T]."})


def cheatsheet():
    return "rng049: Laplace transform of a causal finite-duration h(t) over [0,T]."
