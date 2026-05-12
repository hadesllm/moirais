"""Sample variance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def sample_variance(x, ddof=1, **kwargs) -> DescriptiveResult:
    r"""Compute the sample variance of signal *x*.

    .. math::

        s^2 = \\frac{1}{N - \\text{ddof}} \\sum_{n=0}^{N-1} (x(n) - \\bar{x})^2

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
    var = float(np.var(x, ddof=ddof))
    return DescriptiveResult(
        name="sample_variance",
        value=var,
        extra={"variance": var, "ddof": ddof, "n": len(x)},
    )


svar = sample_variance


def cheatsheet() -> str:
    return "sample_variance({}) -> Sample variance."
