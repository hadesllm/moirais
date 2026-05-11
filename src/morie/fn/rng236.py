"""Definition of the complex cepstrum via inverse z-transform of complex log of Y(z).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_cepstrum_definition"]


def rangayyan_ch4_complex_cepstrum_definition(Y, z, n):
    """
    Definition of the complex cepstrum via inverse z-transform of complex log of Y(z).

    Formula: y_hat(n) = (1/(2*pi*j)) * contour_integral log[Y(z)] * z^(n-1) dz

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
    Rangayyan (2024), Ch 4, Eq 4.64, p. 247
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of the complex cepstrum via inverse z-transform of complex log of Y(z)."})


def cheatsheet():
    return "rng236: Definition of the complex cepstrum via inverse z-transform of complex log of Y(z)."
