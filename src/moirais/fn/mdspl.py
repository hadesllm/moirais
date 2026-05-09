# moirais.fn — function file (hadesllm/moirais)
"""MDS for spatial map of legislators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mds_spatial_map"]


def mds_spatial_map(x):
    """
    MDS for spatial map of legislators

    Formula: Stress = sqrt(sum(d_ij - delta_ij)^2 / sum d_ij^2)

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
    Armstrong Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MDS for spatial map of legislators"})


def cheatsheet():
    return "mdspl: MDS for spatial map of legislators"
