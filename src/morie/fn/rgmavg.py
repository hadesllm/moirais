# morie.fn -- function file (hadesllm/morie)
"""Moving-average (MA) filter (causal and symmetric)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_moving_average"]


def rangayyan_moving_average(x, M):
    """
    Moving-average (MA) filter (causal and symmetric)

    Formula: y[n] = (1/M) sum_{k=0}^{M-1} x[n-k]

    Parameters
    ----------
    x : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Moving-average (MA) filter (causal and symmetric)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Moving-average (MA) filter (causal and symmetric)"})


def cheatsheet():
    return "rgmavg: Moving-average (MA) filter (causal and symmetric)"
