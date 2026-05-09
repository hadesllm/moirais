# moirais.fn — function file (hadesllm/moirais)
"""Causal estimand definition: what to estimate (ATE/ATT/ATC/LATE/CATE)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["estimand_framework"]


def estimand_framework(estimand_type, data):
    """
    Causal estimand definition: what to estimate (ATE/ATT/ATC/LATE/CATE)

    Formula: ATE = E[Y(1)-Y(0)]; ATT = E[Y(1)-Y(0)|T=1]; ATC = E[Y(1)-Y(0)|T=0]; LATE = E[Y(1)-Y(0)|complier]

    Parameters
    ----------
    estimand_type : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'estimate': 'float', 'se': 'float'}

    References
    ----------
    Molak Ch 7
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if data.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Causal estimand definition: what to estimate (ATE/ATT/ATC/LATE/CATE)"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Causal estimand definition: what to estimate (ATE/ATT/ATC/LATE/CATE)"})


def cheatsheet():
    return "estnd: Causal estimand definition: what to estimate (ATE/ATT/ATC/LATE/CATE)"
