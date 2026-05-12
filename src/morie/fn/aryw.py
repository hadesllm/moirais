# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AR model estimation via Yule-Walker equations."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ar_yule_walker_fn(x: np.ndarray, order: int = 10) -> DescriptiveResult:
    """Estimate AR model coefficients using Yule-Walker method.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :return: DescriptiveResult with coefficients and sigma2 in extra.
    """
    from morie._armodel import ar_yule_walker

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = ar_yule_walker(x, order=order)
    return DescriptiveResult(
        name="ar_yule_walker",
        value=None,
        extra={"coefficients": a, "sigma2": float(sigma2)},
    )


aryw = ar_yule_walker_fn


def cheatsheet() -> str:
    return "ar_yule_walker_fn({}) -> AR model estimation via Yule-Walker equations."
