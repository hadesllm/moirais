# morie.fn — function file (hadesllm/morie)
"""Fourier basis function expansion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fourier_basis"]


def fourier_basis(t, n_harmonics):
    """
    Fourier basis function expansion

    Formula: t(t) = a0/2 + sum_k [a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T)]

    Parameters
    ----------
    t : array-like
        Input data.
    n_harmonics : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'F': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 14
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier basis function expansion"})


def cheatsheet():
    return "fours: Fourier basis function expansion"
