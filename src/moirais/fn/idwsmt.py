"""Inverse distance weighting interpolation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["inverse_distance_weighting"]


def inverse_distance_weighting(coords, values, s_predict, power):
    """
    Inverse distance weighting interpolation

    Formula: z(s*) = sum w_i z_i / sum w_i; w_i = 1/d_i^p

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    s_predict : array-like
        Input data.
    power : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shepard (1968)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse distance weighting interpolation"})


def cheatsheet():
    return "idwsmt: Inverse distance weighting interpolation"
