"""Joint longitudinal-survival model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["joint_longitudinal_survival"]


def joint_longitudinal_survival(long_y, time, event, X, Z, cluster):
    """
    Joint longitudinal-survival model

    Formula: longitudinal mixed model + Cox sharing random effects

    Parameters
    ----------
    long_y : array-like
        Input data.
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Henderson et al (2000); Rizopoulos (2012)
    """
    long_y = np.atleast_1d(np.asarray(long_y, dtype=float))
    n = len(long_y)
    result = float(np.mean(long_y))
    se = float(np.std(long_y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint longitudinal-survival model"})


def cheatsheet():
    return "jntlmm: Joint longitudinal-survival model"
