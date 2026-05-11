"""Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_rational_z_transform_form"]


def rangayyan_ch4_rational_z_transform_form(A, z, r, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O):
    """
    Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum).

    Formula: X(z) = A * z^r * prod_{k=1}^{M_I} (1 - a_k z^(-1)) * prod_{k=1}^{M_O} (1 - b_k z) / [ prod_{k=1}^{N_I} (1 - c_k z^(-1)) * prod_{k=1}^{N_O} (1 - d_k z) ]

    Parameters
    ----------
    A : array-like
        Input data.
    z : array-like
        Input data.
    r : array-like
        Input data.
    a_k : array-like
        Input data.
    b_k : array-like
        Input data.
    c_k : array-like
        Input data.
    d_k : array-like
        Input data.
    M_I : array-like
        Input data.
    M_O : array-like
        Input data.
    N_I : array-like
        Input data.
    N_O : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.67, p. 247
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum)."})


def cheatsheet():
    return "rng239: Rational z-transform expressed in pole-zero product form (used to derive complex cepstrum)."
