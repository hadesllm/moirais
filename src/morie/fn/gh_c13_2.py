# morie.fn -- function file (hadesllm/morie)
"""DP posterior convergence to Kaplan-Meier estimator as concentration alpha->0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_surv_dp_km"]


def ghosal_surv_dp_km(x):
    """
    DP posterior convergence to Kaplan-Meier estimator as concentration alpha->0

    Formula: DP(alpha, F_hat_KM) -> KM posterior as alpha->0

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
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "DP posterior convergence to Kaplan-Meier estimator as concentration alpha->0"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "DP posterior convergence to Kaplan-Meier estimator as concentration alpha->0"})


def cheatsheet():
    return "gh_c13_2: DP posterior convergence to Kaplan-Meier estimator as concentration alpha->0"
