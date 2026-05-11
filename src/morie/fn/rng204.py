"""PSD as the Fourier transform of the ACF (Wiener-Khinchin).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_psd_from_acf"]


def rangayyan_ch4_psd_from_acf(phi_xx, X, f, tau):
    """
    PSD as the Fourier transform of the ACF (Wiener-Khinchin).

    Formula: S_xx(f) = FT[phi_xx(tau)] = X(f) * X*(f) = |X(f)|^2

    Parameters
    ----------
    phi_xx : array-like
        Input data.
    X : array-like
        Input data.
    f : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.30, p. 235
    """
    phi_xx = np.atleast_1d(np.asarray(phi_xx, dtype=float))
    n = len(phi_xx)
    result = float(np.mean(phi_xx))
    se = float(np.std(phi_xx, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PSD as the Fourier transform of the ACF (Wiener-Khinchin)."})


def cheatsheet():
    return "rng204: PSD as the Fourier transform of the ACF (Wiener-Khinchin)."
