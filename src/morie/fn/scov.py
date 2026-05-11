# morie.fn — function file (hadesllm/morie)
"""Sample covariance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "A journey of a thousand miles begins with a single step. — Lao Tzu"


def sample_covariance(x, y, **kwargs) -> DescriptiveResult:
    """Compute the sample covariance between *x* and *y*.

    .. math::

        \\text{cov}(x, y) = \\frac{1}{N-1} \\sum_{n=0}^{N-1}
        (x(n) - \\bar{x})(y(n) - \\bar{y})

    Parameters
    ----------
    x, y : array-like
        Input signals of equal length.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    return DescriptiveResult(
        name="sample_covariance",
        value=cov,
        extra={"covariance": cov, "n": len(x)},
    )


scov = sample_covariance


def cheatsheet() -> str:
    return "sample_covariance({}) -> Sample covariance."
