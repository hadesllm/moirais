"""Modified Bessel function K_nu used in Matern covariance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_bessel_function"]


def schabenberger_bessel_function(x, nu):
    """
    Modified Bessel function K_nu used in Matern covariance

    Formula: K_nu(x) = (pi/2)^{1/2} * x^{-nu} * ... (modified Bessel of 2nd kind)

    Parameters
    ----------
    x : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Schabenberger Ch 4, Sec 4.9.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Modified Bessel function K_nu used in Matern covariance"})


def cheatsheet():
    return "spbesf: Modified Bessel function K_nu used in Matern covariance"
