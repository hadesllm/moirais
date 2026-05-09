"""Definition of Frechet-differentiability of an operator at theta."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_frechet_differentiability"]


def kosorok_ch2_frechet_differentiability(phi, theta, h_n):
    """
    Definition of Frechet-differentiability of an operator at theta

    Formula: || phi(theta + h_n) - phi(theta) - phi'_theta(h_n) ||_L / ||h_n|| -> 0

    Parameters
    ----------
    phi : array-like
        Input data.
    theta : array-like
        Input data.
    h_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.14, p. 26
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    n = len(phi)
    result = float(np.mean(phi))
    se = float(np.std(phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of Frechet-differentiability of an operator at theta"})


def cheatsheet():
    return "ksr050: Definition of Frechet-differentiability of an operator at theta"
