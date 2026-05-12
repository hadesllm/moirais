# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Backward finite difference."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It's a trap!"


def backward_difference(x, order=1, **kwargs) -> DescriptiveResult:
    """Compute backward finite difference of signal *x*.

    The backward difference at index i is x[i] - x[i-1].

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
        d = d[1:] - d[:-1]
    return DescriptiveResult(
        name="backward_difference",
        value=float(np.mean(np.abs(d))),
        extra={
            "order": order,
            "length": len(d),
            "mean_abs": float(np.mean(np.abs(d))),
            "max_abs": float(np.max(np.abs(d))),
        },
    )


bkdif = backward_difference


def cheatsheet() -> str:
    return "backward_difference({}) -> Backward finite difference."
