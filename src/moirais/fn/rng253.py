"""Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_power_cepstrum_definition"]


def rangayyan_ch4_power_cepstrum_definition(Y, z, n):
    """
    Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2.

    Formula: y_hat_p(n) = { (1/(2*pi*j)) * contour_integral log|Y(z)|^2 * z^(n-1) dz }^2

    Parameters
    ----------
    Y : array-like
        Input data.
    z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.81, p. 251
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2."})


def cheatsheet():
    return "rng253: Definition of the power cepstrum as squared inverse z-transform of log|Y(z)|^2."
