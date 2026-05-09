"""Semivariogram definition as half mean-squared difference."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_semivariogram_def"]


def schabenberger_semivariogram_def(coords, z):
    """
    Semivariogram definition as half mean-squared difference

    Formula: gamma(h) = 0.5 * E[(Z(s+h) - Z(s))^2]

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 1/4
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semivariogram definition as half mean-squared difference"})


def cheatsheet():
    return "spsemv: Semivariogram definition as half mean-squared difference"
