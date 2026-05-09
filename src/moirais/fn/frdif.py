# moirais.fn — function file (hadesllm/moirais)
"""Forward finite difference."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def forward_difference(x, order=1, **kwargs) -> DescriptiveResult:
    """Compute forward finite difference of signal *x*.

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
        d = np.diff(d)
    return DescriptiveResult(
        name="forward_difference",
        value=float(np.mean(np.abs(d))),
        extra={
            "order": order,
            "length": len(d),
            "mean_abs": float(np.mean(np.abs(d))),
            "max_abs": float(np.max(np.abs(d))),
        },
    )


frdif = forward_difference


def cheatsheet() -> str:
    return "forward_difference({}) -> Forward finite difference."
