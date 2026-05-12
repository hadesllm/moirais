# morie.fn -- function file (hadesllm/morie)
"""Central finite difference."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is another."


def central_difference(x, order=1, **kwargs) -> DescriptiveResult:
    """Compute central finite difference of signal *x*.

    For order 1: d[i] = (x[i+1] - x[i-1]) / 2

    Parameters
    ----------
    x : array-like
        Input signal.
    order : int
        Difference order (default 1).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    d = x.copy()
    for _ in range(order):
        d = (d[2:] - d[:-2]) / 2.0
    return DescriptiveResult(
        name="central_difference",
        value=float(np.mean(np.abs(d))),
        extra={
            "order": order,
            "length": len(d),
            "mean_abs": float(np.mean(np.abs(d))),
            "max_abs": float(np.max(np.abs(d))),
        },
    )


cntdf = central_difference


def cheatsheet() -> str:
    return "central_difference({}) -> Central finite difference."
