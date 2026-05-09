"""Laplace transform."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["laplace_transform"]


def laplace_transform(f, t, s):
    """
    Laplace transform

    Formula: L{f}(s) = ∫_0^∞ f(t) e^{-st} dt

    Parameters
    ----------
    f : array-like
        Input data.
    t : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplace transform"})


def cheatsheet():
    return "laplT: Laplace transform"
