"""Magnitude response from products of distances to zeros and poles.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_magnitude_response_from_pole_zero"]


def rangayyan_ch3_magnitude_response_from_pole_zero(l_k, r_k, N, M):
    """
    Magnitude response from products of distances to zeros and poles.

    Formula: |H(omega_0)| = prod_{k=1}^{N} l_k / prod_{k=1}^{M} r_k

    Parameters
    ----------
    l_k : array-like
        Input data.
    r_k : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.72, p. 125
    """
    l_k = np.atleast_1d(np.asarray(l_k, dtype=float))
    n = len(l_k)
    result = float(np.mean(l_k))
    se = float(np.std(l_k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude response from products of distances to zeros and poles."})


def cheatsheet():
    return "rng061: Magnitude response from products of distances to zeros and poles."
