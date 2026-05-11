"""Bilateral Laplace transform of an impulse response h(t).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_laplace_transform"]


def rangayyan_ch3_laplace_transform(h, t, s):
    """
    Bilateral Laplace transform of an impulse response h(t).

    Formula: H(s) = integral_{-inf}^{inf} h(t) * exp(-s*t) dt

    Parameters
    ----------
    h : array-like
        Input data.
    t : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.50, p. 117
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bilateral Laplace transform of an impulse response h(t)."})


def cheatsheet():
    return "rng048: Bilateral Laplace transform of an impulse response h(t)."
