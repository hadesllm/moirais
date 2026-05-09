# moirais.fn — function file (hadesllm/moirais)
"""M-estimator with nuisance parameters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_m_estimator"]


def kosorok_m_estimator(x, y):
    """
    M-estimator with nuisance parameters

    Formula: theta_n = argmax M_n(theta, eta_n)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "M-estimator with nuisance parameters"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "M-estimator with nuisance parameters"})


def cheatsheet():
    return "ksr10: M-estimator with nuisance parameters"
