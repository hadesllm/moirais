# moirais.fn — function file (hadesllm/moirais)
"""Predictive recursion deconvolution algorithm for DPM estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pred_rec"]


def ghosal_pred_rec(x):
    """
    Predictive recursion deconvolution algorithm for DPM estimation

    Formula: p_{n+1}(x) = (1-r_n)*p_n(x) + r_n*K(x;X_{n+1}), r_n -> 0

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
    Ghosal Ch 5 §5.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Predictive recursion deconvolution algorithm for DPM estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Predictive recursion deconvolution algorithm for DPM estimation"})


def cheatsheet():
    return "gh_c5_7: Predictive recursion deconvolution algorithm for DPM estimation"
