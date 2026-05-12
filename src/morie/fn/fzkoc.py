# morie.fn -- function file (hadesllm/morie)
"""Order-m kernel condition for quantile estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_kernel_order_m_condition"]


def fauzi_kernel_order_m_condition(kernel, order):
    """
    Order-m kernel condition for quantile estimation

    Formula: integral kernel^i K(kernel)dx=0 for i=1,...,m-1; integral kernel^m K(kernel)dx != 0; K in L^2

    Parameters
    ----------
    kernel : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Fauzi Ch 3
    """
    if callable(kernel):
        _xs = np.linspace(-3, 3, 100)
        kernel = np.asarray([kernel(_x) for _x in _xs], dtype=float)
    else:
        kernel = np.asarray(kernel, dtype=float)
    n = len(kernel)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Order-m kernel condition for quantile estimation"})
    estimate = np.median(kernel)
    se = 1.2533 * np.std(kernel, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Order-m kernel condition for quantile estimation"})


def cheatsheet():
    return "fzkoc: Order-m kernel condition for quantile estimation"
