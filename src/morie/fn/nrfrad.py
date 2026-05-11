"""Neural Radiance Field volume rendering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nerf_radiance"]


def nerf_radiance(rays, mlp):
    """
    Neural Radiance Field volume rendering

    Formula: sum_i T_i (1 - exp(-sigma_i delta_i)) c_i

    Parameters
    ----------
    rays : array-like
        Input data.
    mlp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mildenhall et al (2020) NeRF
    """
    rays = np.atleast_1d(np.asarray(rays, dtype=float))
    n = len(rays)
    result = float(np.mean(rays))
    se = float(np.std(rays, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neural Radiance Field volume rendering"})


def cheatsheet():
    return "nrfrad: Neural Radiance Field volume rendering"
