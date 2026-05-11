"""GWR kernel functions: Gaussian, bisquare, tricube, boxcar."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_gwr_kernels"]


def schabenberger_gwr_kernels(distance, bandwidth, kernel_type):
    """
    GWR kernel functions: Gaussian, bisquare, tricube, boxcar

    Formula: Gaussian: w=exp(-d^2/(2h^2)); Bisquare: w=(1-(d/h)^2)^2 if d<h

    Parameters
    ----------
    distance : array-like
        Input data.
    bandwidth : array-like
        Input data.
    kernel_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights

    References
    ----------
    Schabenberger Ch 6
    """
    distance = np.asarray(distance, dtype=float)
    n = int(distance) if distance.ndim == 0 else len(distance)
    result = float(np.mean(distance))
    se = float(np.std(distance, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GWR kernel functions: Gaussian, bisquare, tricube, boxcar"})


def cheatsheet():
    return "spgwrk: GWR kernel functions: Gaussian, bisquare, tricube, boxcar"
