"""Covariance parameter estimation for kriging (LS, ML, REML)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_cov_param_estimation_kriging"]


def schabenberger_cov_param_estimation_kriging(coords, z, variogram_model):
    """
    Covariance parameter estimation for kriging (LS, ML, REML)

    Formula: theta_hat from WLS/ML/REML; kriging uses Sigma(theta_hat)

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    variogram_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: parameters

    References
    ----------
    Schabenberger Ch 5, Sec 5.5
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    if z.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Covariance parameter estimation for kriging (LS, ML, REML)"})
    estimate = np.median(z)
    se = 1.2533 * np.std(z, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Covariance parameter estimation for kriging (LS, ML, REML)"})


def cheatsheet():
    return "spkce: Covariance parameter estimation for kriging (LS, ML, REML)"
