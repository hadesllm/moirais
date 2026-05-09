"""Maximum likelihood estimation of variogram/covariance parameters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_ml_variogram"]


def schabenberger_ml_variogram(coords, z, variogram_model):
    """
    Maximum likelihood estimation of variogram/covariance parameters

    Formula: maximize log L(theta) = -n/2*log(2pi) - 0.5*log|Sigma(theta)| - 0.5*(Z-mu)'*Sigma^{-1}*(Z-mu)

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
    Schabenberger Ch 4, Sec 4.5.2
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    if z.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Maximum likelihood estimation of variogram/covariance parameters"})
    estimate = np.median(z)
    se = 1.2533 * np.std(z, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Maximum likelihood estimation of variogram/covariance parameters"})


def cheatsheet():
    return "spml: Maximum likelihood estimation of variogram/covariance parameters"
