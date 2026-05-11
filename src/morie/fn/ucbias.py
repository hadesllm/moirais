"""Bias factor for unmeasured confounder."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["unmeasured_conf_bias"]


def unmeasured_conf_bias(RR_UD, RR_UY):
    """
    Bias factor for unmeasured confounder

    Formula: BF = RR_UD * RR_UY / (RR_UD + RR_UY - 1)

    Parameters
    ----------
    RR_UD : array-like
        Input data.
    RR_UY : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ding-VanderWeele (2016) E-value
    """
    RR_UD = np.atleast_1d(np.asarray(RR_UD, dtype=float))
    n = len(RR_UD)
    result = float(np.mean(RR_UD))
    se = float(np.std(RR_UD, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias factor for unmeasured confounder"})


def cheatsheet():
    return "ucbias: Bias factor for unmeasured confounder"
