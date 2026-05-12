# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AR model estimation via covariance method."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience, there is no such thing as luck."


def ar_covariance_fn(x: np.ndarray, order: int = 4) -> DescriptiveResult:
    """Estimate AR model coefficients using the covariance method.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :return: DescriptiveResult with coefficients and sigma2 in extra.
    """
    from morie._armodel import ar_covariance

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = ar_covariance(x, order=order)
    return DescriptiveResult(
        name="ar_covariance",
        value=None,
        extra={"coefficients": a, "sigma2": float(sigma2)},
    )


arcov = ar_covariance_fn


def cheatsheet() -> str:
    return "ar_covariance_fn({}) -> AR model estimation via covariance method."
