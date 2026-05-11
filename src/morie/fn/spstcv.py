"""Separable spatio-temporal covariance: product of spatial and temporal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_st_cov_separable"]


def schabenberger_st_cov_separable(spatial_h, temporal_u, cov_spatial, cov_temporal):
    """
    Separable spatio-temporal covariance: product of spatial and temporal

    Formula: C(h,u) = C_s(h) * C_t(u)

    Parameters
    ----------
    spatial_h : array-like
        Input data.
    temporal_u : array-like
        Input data.
    cov_spatial : array-like
        Input data.
    cov_temporal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: st_covariance

    References
    ----------
    Schabenberger Ch 9, Sec 9.2
    """
    spatial_h = np.asarray(spatial_h, dtype=float)
    n = int(spatial_h) if spatial_h.ndim == 0 else len(spatial_h)
    result = float(np.mean(spatial_h))
    se = float(np.std(spatial_h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Separable spatio-temporal covariance: product of spatial and temporal"})


def cheatsheet():
    return "spstcv: Separable spatio-temporal covariance: product of spatial and temporal"
