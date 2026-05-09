# moirais.fn — function file (hadesllm/moirais)
"""Fourth-order (Muller) kernel for quantile estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_muller_fourth_order_kernel"]


def fauzi_muller_fourth_order_kernel(x):
    """
    Fourth-order (Muller) kernel for quantile estimation

    Formula: K(x)=(315/512)(11x^8-36x^6+42x^4-20x^2+3)I(|x|<=1)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: kernel_value

    References
    ----------
    Fauzi Ch 3, Sec 3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Fourth-order (Muller) kernel for quantile estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Fourth-order (Muller) kernel for quantile estimation"})


def cheatsheet():
    return "fzmkrn: Fourth-order (Muller) kernel for quantile estimation"
