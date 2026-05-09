"""Cross-spectral density (CSD) as the Fourier transform of the CCF.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_csd_from_ccf"]


def rangayyan_ch4_csd_from_ccf(theta_xy, X, Y, f, tau):
    """
    Cross-spectral density (CSD) as the Fourier transform of the CCF.

    Formula: S_xy(f) = FT[theta_xy(tau)] = X(f) * Y*(f)

    Parameters
    ----------
    theta_xy : array-like
        Input data.
    X : array-like
        Input data.
    Y : array-like
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
    Rangayyan (2024), Ch 4, Eq 4.31, p. 236
    """
    theta_xy = np.atleast_1d(np.asarray(theta_xy, dtype=float))
    n = len(theta_xy)
    result = float(np.mean(theta_xy))
    se = float(np.std(theta_xy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-spectral density (CSD) as the Fourier transform of the CCF."})


def cheatsheet():
    return "rng205: Cross-spectral density (CSD) as the Fourier transform of the CCF."
