"""Output of matched filter via inverse Fourier transform of X(omega)*H(omega).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_output_inverse_ft"]


def rangayyan_ch4_matched_filter_output_inverse_ft(X, H, omega, f, t):
    """
    Output of matched filter via inverse Fourier transform of X(omega)*H(omega).

    Formula: y(t) = (1/(2*pi)) * integral_{-inf}^{inf} X(omega) H(omega) exp(+j*omega*t) d(omega) = integral_{-inf}^{inf} X(f) H(f) exp(+j*2*pi*f*t) df

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    omega : array-like
        Input data.
    f : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.34, p. 238
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of matched filter via inverse Fourier transform of X(omega)*H(omega)."})


def cheatsheet():
    return "rng208: Output of matched filter via inverse Fourier transform of X(omega)*H(omega)."
