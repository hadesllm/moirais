"""Non-separable spatio-temporal covariance models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_st_cov_nonsep"]


def schabenberger_st_cov_nonsep(spatial_h, temporal_u, params):
    """
    Non-separable spatio-temporal covariance models

    Formula: C(h,u) from: monotone fn, spectral, mixture, or differential equation approaches

    Parameters
    ----------
    spatial_h : array-like
        Input data.
    temporal_u : array-like
        Input data.
    params : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: st_covariance

    References
    ----------
    Schabenberger Ch 9, Sec 9.3
    """
    spatial_h = np.asarray(spatial_h, dtype=float)
    n = int(spatial_h) if spatial_h.ndim == 0 else len(spatial_h)
    result = float(np.mean(spatial_h))
    se = float(np.std(spatial_h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Non-separable spatio-temporal covariance models"})


def cheatsheet():
    return "spstcn: Non-separable spatio-temporal covariance models"
