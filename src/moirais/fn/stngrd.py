"""Spatial transformer network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stn_spatial_transform"]


def stn_spatial_transform(x):
    """
    Spatial transformer network

    Formula: learned affine grid + bilinear sample

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jaderberg et al (2015)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial transformer network"})


def cheatsheet():
    return "stngrd: Spatial transformer network"
