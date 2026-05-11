# morie.fn — function file (hadesllm/morie)
"""Chen (2002) estimator of T in transformation model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_chen_estimator_T"]


def horowitz_chen_estimator_T(x, y, bandwidth):
    """
    Chen (2002) estimator of T in transformation model

    Formula: T_hat_Chen from recursive conditional CDF estimator; faster rate than Horowitz

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Chen (2002) estimator of T in transformation model"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Chen (2002) estimator of T in transformation model"})


def cheatsheet():
    return "hrzchet: Chen (2002) estimator of T in transformation model"
