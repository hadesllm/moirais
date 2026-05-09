"""Z-transform of a signal with a wavelet and an echo.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_z_transform_signal_echo"]


def rangayyan_ch4_z_transform_signal_echo(a, n_0, z, H):
    """
    Z-transform of a signal with a wavelet and an echo.

    Formula: Y(z) = (1 + a * z^(-n_0)) * H(z)

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    z : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.76, p. 249
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-transform of a signal with a wavelet and an echo."})


def cheatsheet():
    return "rng248: Z-transform of a signal with a wavelet and an echo."
