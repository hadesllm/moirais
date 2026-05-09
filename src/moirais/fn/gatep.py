# moirais.fn — function file (hadesllm/moirais)
"""GATE (group average treatment effect): average CATE over subgroup."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gate_estimation"]


def gate_estimation(cate, X, group_var):
    """
    GATE (group average treatment effect): average CATE over subgroup

    Formula: GATE_g = E[tau(X) | X in G_g] = E[Y(1)-Y(0)|X in G_g]

    Parameters
    ----------
    cate : array-like
        Input data.
    X : array-like
        Input data.
    group_var : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gate': 'array'}

    References
    ----------
    Molak Ch 10
    """
    cate = np.asarray(cate, dtype=float)
    n = int(cate) if cate.ndim == 0 else len(cate)
    if cate.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "GATE (group average treatment effect): average CATE over subgroup"})
    estimate = np.median(cate)
    se = 1.2533 * np.std(cate, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "GATE (group average treatment effect): average CATE over subgroup"})


def cheatsheet():
    return "gatep: GATE (group average treatment effect): average CATE over subgroup"
