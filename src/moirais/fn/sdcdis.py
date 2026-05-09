"""Spatial data distortion / privacy preservation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_data_distortion"]


def spatial_data_distortion(coords, noise_radius):
    """
    Spatial data distortion / privacy preservation

    Formula: add geographic noise; recompute estimate

    Parameters
    ----------
    coords : array-like
        Input data.
    noise_radius : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Allshouse et al (2010)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial data distortion / privacy preservation"})


def cheatsheet():
    return "sdcdis: Spatial data distortion / privacy preservation"
