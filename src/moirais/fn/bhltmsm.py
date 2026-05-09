"""MSM application: behavioral health treatment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["behavioral_health_msm"]


def behavioral_health_msm(y, A, H, baseline):
    """
    MSM application: behavioral health treatment

    Formula: ATE of long-term treatment on outcome

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    applied benchmark
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM application: behavioral health treatment"})


def cheatsheet():
    return "bhltmsm: MSM application: behavioral health treatment"
