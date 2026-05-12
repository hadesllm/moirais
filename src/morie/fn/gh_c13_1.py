# morie.fn -- function file (hadesllm/morie)
"""DP prior for survival: Susarla-Van Ryzin posterior under right censoring."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_surv_dp_post"]


def ghosal_surv_dp_post(x):
    """
    DP prior for survival: Susarla-Van Ryzin posterior under right censoring

    Formula: F|X ~ DP posterior accounting for censored observations

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 13 §13.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP prior for survival: Susarla-Van Ryzin posterior under right censoring"})


def cheatsheet():
    return "gh_c13_1: DP prior for survival: Susarla-Van Ryzin posterior under right censoring"
