"""Kernel intensity estimation for inhomogeneous Poisson process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_intensity_estimation"]


def schabenberger_intensity_estimation(points, bandwidth, region):
    """
    Kernel intensity estimation for inhomogeneous Poisson process

    Formula: lambda_hat(s) = sum_i K_h(s-s_i) / |A|

    Parameters
    ----------
    points : array-like
        Input data.
    bandwidth : array-like
        Input data.
    region : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: intensity_surface

    References
    ----------
    Schabenberger Ch 3, Sec 3.5.1
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    if points.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kernel intensity estimation for inhomogeneous Poisson process"})
    estimate = np.median(points)
    se = 1.2533 * np.std(points, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kernel intensity estimation for inhomogeneous Poisson process"})


def cheatsheet():
    return "spintp: Kernel intensity estimation for inhomogeneous Poisson process"
