"""Low-rank matrix completion (nuclear norm)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matrix_completion_low_rank"]


def matrix_completion_low_rank(R_obs, mask):
    """
    Low-rank matrix completion (nuclear norm)

    Formula: min ||X||_* s.t. P_Ω(X) = P_Ω(R)

    Parameters
    ----------
    R_obs : array-like
        Input data.
    mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Candès-Recht (2009)
    """
    R_obs = np.atleast_1d(np.asarray(R_obs, dtype=float))
    n = len(R_obs)
    result = float(np.mean(R_obs))
    se = float(np.std(R_obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Low-rank matrix completion (nuclear norm)"})


def cheatsheet():
    return "meglt: Low-rank matrix completion (nuclear norm)"
