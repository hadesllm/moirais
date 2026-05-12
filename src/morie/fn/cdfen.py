# morie.fn -- function file (hadesllm/morie)
"""Empirical CDF estimation."""

import numpy as np

from ._containers import DescriptiveResult
def cdf_estimate(x, **kwargs) -> DescriptiveResult:
    r"""
    Compute the empirical cumulative distribution function (ECDF).

    .. math::

        \\hat{F}(t) = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbf{1}(x_i \\le t)

    :param x: array-like of observations.
    :return: DescriptiveResult with sorted values and ECDF probabilities.

    References
    ----------
    van der Vaart AW (1998). Asymptotic Statistics.
    Cambridge University Press.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 1:
        raise ValueError("Need at least 1 observation.")
    xs = np.sort(x)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    return DescriptiveResult(
        name="cdf_estimate",
        value=float(np.median(xs)),
        extra={
            "x_sorted": xs.tolist(),
            "ecdf": ecdf.tolist(),
            "n": len(x),
        },
    )


cdfen = cdf_estimate


def cheatsheet() -> str:
    return "cdf_estimate({}) -> Empirical CDF estimation."
