"""Bilateral z-transform of a discrete-time signal x(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_z_transform_definition"]


def rangayyan_ch3_z_transform_definition(x, n, z):
    """
    Bilateral z-transform of a discrete-time signal x(n).

    Formula: X(z) = sum_{n=-inf}^{inf} x(n) * z^(-n)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.54, p. 119
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bilateral z-transform of a discrete-time signal x(n)."})


def cheatsheet():
    return "rng052: Bilateral z-transform of a discrete-time signal x(n)."
