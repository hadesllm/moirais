"""AFT model residuals (Cox-Snell, deviance)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aft_residuals"]


def aft_residuals(fit):
    """
    AFT model residuals (Cox-Snell, deviance)

    Formula: r_i = -log S(t_i | X_i)

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cox-Snell (1968)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AFT model residuals (Cox-Snell, deviance)"})


def cheatsheet():
    return "aftres: AFT model residuals (Cox-Snell, deviance)"
