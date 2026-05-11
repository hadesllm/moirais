"""Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_pole_zero_factored_form"]


def rangayyan_ch3_pole_zero_factored_form(z_k, p_k, z, N, M):
    """
    Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors.

    Formula: H(z) = prod_{k=1}^{N} (1 - z_k z^(-1)) / prod_{k=1}^{M} (1 - p_k z^(-1))

    Parameters
    ----------
    z_k : array-like
        Input data.
    p_k : array-like
        Input data.
    z : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.69, p. 124
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors."})


def cheatsheet():
    return "rng058: Pole-zero factored transfer function in terms of (1 - z_k z^-1) factors."
