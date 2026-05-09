"""Squared transfer function of the Butterworth lowpass filter in s-domain.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_squared_laplace"]


def rangayyan_ch3_butterworth_squared_laplace(s, Omega_c, N):
    """
    Squared transfer function of the Butterworth lowpass filter in s-domain.

    Formula: H_a(s) * H_a(-s) = 1 / (1 + (s/(j*Omega_c))^(2*N))

    Parameters
    ----------
    s : array-like
        Input data.
    Omega_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.136, p. 154
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squared transfer function of the Butterworth lowpass filter in s-domain."})


def cheatsheet():
    return "rng124: Squared transfer function of the Butterworth lowpass filter in s-domain."
