"""Sample standard deviation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truth comes out of error more readily than out of confusion. — Francis Bacon"


def sample_std(x, ddof=1, **kwargs) -> DescriptiveResult:
    r"""Compute the sample standard deviation of signal *x*.

    .. math::

        s = \\sqrt{\\frac{1}{N - \\text{ddof}} \\sum_{n=0}^{N-1} (x(n) - \\bar{x})^2}

    Parameters
    ----------
    x : array-like
        Input signal or data vector.
    ddof : int
        Delta degrees of freedom (default 1 for unbiased).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x, ddof=ddof))
    return DescriptiveResult(
        name="sample_std",
        value=sd,
        extra={"std": sd, "ddof": ddof, "n": len(x)},
    )


sstd = sample_std


def cheatsheet() -> str:
    return "sample_std({}) -> Sample standard deviation."
