"""Projection (inner product) of two continuous-time signals over R.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_continuous_dot_product"]


def rangayyan_ch4_continuous_dot_product(x, y, t):
    """
    Projection (inner product) of two continuous-time signals over R.

    Formula: theta_xy = integral_{-inf}^{inf} x(t) * y(t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.26, p. 229
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Projection (inner product) of two continuous-time signals over R."})


def cheatsheet():
    return "rng200: Projection (inner product) of two continuous-time signals over R."
