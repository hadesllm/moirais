# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""AR model estimation via Burg algorithm."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius"


def ar_burg_fn(x: np.ndarray, order: int = 4) -> DescriptiveResult:
    """Estimate AR model coefficients using the Burg algorithm.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :return: DescriptiveResult with coefficients and sigma2 in extra.
    """
    from morie._armodel import ar_burg

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = ar_burg(x, order=order)
    return DescriptiveResult(
        name="ar_burg",
        value=None,
        extra={"coefficients": a, "sigma2": float(sigma2)},
    )


arbrg = ar_burg_fn


def cheatsheet() -> str:
    return "ar_burg_fn({}) -> AR model estimation via Burg algorithm."
