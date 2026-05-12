"""Sample mean."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience, there is no such thing as luck."


def sample_mean(x, **kwargs) -> DescriptiveResult:
    r"""Compute the sample mean of signal *x*.

    .. math::

        \\bar{x} = \\frac{1}{N} \\sum_{n=0}^{N-1} x(n)

    Parameters
    ----------
    x : array-like
        Input signal or data vector.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    return DescriptiveResult(
        name="sample_mean",
        value=mu,
        extra={"mean": mu, "n": len(x)},
    )


smean = sample_mean


def cheatsheet() -> str:
    return "sample_mean({}) -> Sample mean."
