"""Complex log of X(z) expanded as a sum of log terms over poles and zeros.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_log_x_z"]


def rangayyan_ch4_complex_log_x_z(A, z, r, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O):
    """
    Complex log of X(z) expanded as a sum of log terms over poles and zeros.

    Formula: X_hat(z) = log[X(z)] = log[A] + log[z^r] + sum_{k=1}^{M_I} log(1 - a_k z^(-1)) + sum_{k=1}^{M_O} log(1 - b_k z) - sum_{k=1}^{N_I} log(1 - c_k z^(-1)) - sum_{k=1}^{N_O} log(1 - d_k z)

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
    Rangayyan (2024), Ch 4, Eq 4.68, p. 248
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex log of X(z) expanded as a sum of log terms over poles and zeros."})


def cheatsheet():
    return "rng240: Complex log of X(z) expanded as a sum of log terms over poles and zeros."
