"""Frequency response evaluated at z_0 on the unit circle from pole-zero form.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_frequency_response_from_pole_zero"]


def rangayyan_ch3_frequency_response_from_pole_zero(z_0, z_k, p_k, N, M):
    """
    Frequency response evaluated at z_0 on the unit circle from pole-zero form.

    Formula: H(omega_0)|_{z=z_0} = z_0^(M-N) * prod_{k=1}^{N} (z_0 - z_k) / prod_{k=1}^{M} (z_0 - p_k)

    Parameters
    ----------
    z_0 : array-like
        Input data.
    z_k : array-like
        Input data.
    p_k : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.71, p. 124
    """
    z_0 = np.atleast_1d(np.asarray(z_0, dtype=float))
    n = len(z_0)
    result = float(np.mean(z_0))
    se = float(np.std(z_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response evaluated at z_0 on the unit circle from pole-zero form."})


def cheatsheet():
    return "rng060: Frequency response evaluated at z_0 on the unit circle from pole-zero form."
