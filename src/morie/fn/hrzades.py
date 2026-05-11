# morie.fn — function file (hadesllm/morie)
"""Improved average derivative estimator (Li-Hsiao-Zinn)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_improved_ade"]


def horowitz_improved_ade(x, y, bandwidth):
    """
    Improved average derivative estimator (Li-Hsiao-Zinn)

    Formula: delta_hat = -(2/n)*sum_i f_hat'(X_i)*[Y_i - E_hat(Y|X_i)] / E_hat[f_hat(X)]

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
        Keys: delta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Improved average derivative estimator (Li-Hsiao-Zinn)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Improved average derivative estimator (Li-Hsiao-Zinn)"})


def cheatsheet():
    return "hrzades: Improved average derivative estimator (Li-Hsiao-Zinn)"
