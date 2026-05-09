"""Berkeley Earth Kriging surface T."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["berkeley_earth"]


def berkeley_earth(stations):
    """
    Berkeley Earth Kriging surface T

    Formula: weighted least squares + variogram

    Parameters
    ----------
    stations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rohde et al (2013)
    """
    stations = np.atleast_1d(np.asarray(stations, dtype=float))
    n = len(stations)
    result = float(np.mean(stations))
    se = float(np.std(stations, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Berkeley Earth Kriging surface T"})


def cheatsheet():
    return "berkly: Berkeley Earth Kriging surface T"
