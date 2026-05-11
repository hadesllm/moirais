# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""AR model estimation via Yule-Walker (autocorrelation method)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The road up and the road down are the same thing. — Heraclitus"


def ar_yule_walker_fn(x: np.ndarray, order: int = 4) -> DescriptiveResult:
    """Estimate AR coefficients via the Yule-Walker (autocorrelation) method.

    Solves the Yule-Walker equations :math:`R a = r` using Levinson-Durbin
    recursion, where *R* is the Toeplitz autocorrelation matrix.

    :param x: 1-D input signal.
    :param order: AR model order (default 4).
    :return: DescriptiveResult with coefficients and prediction error variance.
    """
    from morie._armodel import ar_yule_walker

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = ar_yule_walker(x, order=order)
    return DescriptiveResult(
        name="ar_yule_walker",
        value=float(sigma2),
        extra={"coefficients": a, "sigma2": float(sigma2), "order": order},
    )


arywk = ar_yule_walker_fn


def cheatsheet() -> str:
    return "ar_yule_walker_fn({}) -> AR model estimation via Yule-Walker (autocorrelation method)"
