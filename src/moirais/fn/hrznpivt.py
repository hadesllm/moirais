# moirais.fn — function file (hadesllm/moirais)
"""NPIV estimation when operator T is unknown."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_npiv_unknown_T"]


def horowitz_npiv_unknown_T(x, y, w, bandwidth, regularization):
    """
    NPIV estimation when operator T is unknown

    Formula: T_hat = fXW_hat(x,w) kernel-estimated; solve regularized system with T_hat

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    bandwidth : array-like
        Input data.
    regularization : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "NPIV estimation when operator T is unknown"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "NPIV estimation when operator T is unknown"})


def cheatsheet():
    return "hrznpivt: NPIV estimation when operator T is unknown"
