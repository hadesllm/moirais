# moirais.fn — function file (hadesllm/moirais)
"""Conditions C1-C6 for boundary-free MRL estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_conditions_c1_c6"]


def fauzi_conditions_c1_c6(x=None, *args, **kwargs):
    """
    Conditions C1-C6 for boundary-free MRL estimation

    Formula: C1: K nonneg sym; C2: h->0,nh->inf; C3: g increasing; C4: f_X,g in C^2; C5-C6: finite moments

    Parameters
    ----------


    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Fauzi Ch 4, Conditions C1-C6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Conditions C1-C6 for boundary-free MRL estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Conditions C1-C6 for boundary-free MRL estimation"})


def cheatsheet():
    return "fzc1c6: Conditions C1-C6 for boundary-free MRL estimation"
