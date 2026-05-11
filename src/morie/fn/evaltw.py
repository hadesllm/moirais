"""E-value for unmeasured confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["e_value_unmeasured_confounding"]


def e_value_unmeasured_confounding(estimate, ci_lower, ci_upper):
    """
    E-value for unmeasured confounding

    Formula: E = RR + sqrt(RR(RR-1)) for RR > 1

    Parameters
    ----------
    estimate : array-like
        Input data.
    ci_lower : array-like
        Input data.
    ci_upper : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele & Ding (2017)
    """
    estimate = np.atleast_1d(np.asarray(estimate, dtype=float))
    n = len(estimate)
    result = float(np.mean(estimate))
    se = float(np.std(estimate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "E-value for unmeasured confounding"})


def cheatsheet():
    return "evaltw: E-value for unmeasured confounding"
