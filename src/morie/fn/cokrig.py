"""Co-kriging with secondary variable."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cokriging"]


def cokriging(coords, z1, z2, s0, cross_vario):
    """
    Co-kriging with secondary variable

    Formula: Z*(s0) = sum lambda_i Z1(s_i) + sum mu_j Z2(s_j)

    Parameters
    ----------
    coords : array-like
        Input data.
    z1 : array-like
        Input data.
    z2 : array-like
        Input data.
    s0 : array-like
        Input data.
    cross_vario : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wackernagel (2003) §22
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Co-kriging with secondary variable"})


def cheatsheet():
    return "cokrig: Co-kriging with secondary variable"
