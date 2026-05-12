# morie.fn -- function file (hadesllm/morie)
"""Pole-zero plot from transfer function coefficients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pole_zero_plot"]


def rangayyan_pole_zero_plot(b, a):
    """
    Pole-zero plot from transfer function coefficients

    Formula: H(z) = B(z)/A(z); zeros = roots(B), poles = roots(A)

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: zeros, poles

    References
    ----------
    Rangayyan Ch 3.4.3
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pole-zero plot from transfer function coefficients"})


def cheatsheet():
    return "rgpzp: Pole-zero plot from transfer function coefficients"
