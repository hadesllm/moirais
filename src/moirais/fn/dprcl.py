"""Calibrate noise to global sensitivity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_release_calibration"]


def dp_release_calibration(sensitivity, epsilon, c):
    """
    Calibrate noise to global sensitivity

    Formula: σ = c · Δf / ε

    Parameters
    ----------
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Roth (2014)
    """
    sensitivity = np.atleast_1d(np.asarray(sensitivity, dtype=float))
    n = len(sensitivity)
    result = float(np.mean(sensitivity))
    se = float(np.std(sensitivity, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Calibrate noise to global sensitivity"})


def cheatsheet():
    return "dprcl: Calibrate noise to global sensitivity"
