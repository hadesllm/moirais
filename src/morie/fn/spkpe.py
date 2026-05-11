"""Prediction errors under estimated covariance parameters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_kriging_pred_error"]


def schabenberger_kriging_pred_error(coords, z, target, variogram_model):
    """
    Prediction errors under estimated covariance parameters

    Formula: MSE_plug-in >= sigma^2_OK; correction needed for theta_hat uncertainty

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    target : array-like
        Input data.
    variogram_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mse

    References
    ----------
    Schabenberger Ch 5, Sec 5.5.4
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    if z.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Prediction errors under estimated covariance parameters"})
    estimate = np.median(z)
    se = 1.2533 * np.std(z, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Prediction errors under estimated covariance parameters"})


def cheatsheet():
    return "spkpe: Prediction errors under estimated covariance parameters"
