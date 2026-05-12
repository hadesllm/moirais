"""Skewness coefficient."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def skewness_coeff(x, **kwargs) -> DescriptiveResult:
    r"""Compute the skewness coefficient of signal *x*.

    .. math::

        \\gamma_1 = \\frac{\\mu_3}{\\sigma^3}

    where :math:`\\mu_3` is the third central moment.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)
    if sigma == 0.0:
        skew = 0.0
    else:
        mu3 = float(np.mean((x - mu) ** 3))
        skew = mu3 / (sigma**3)
    return DescriptiveResult(
        name="skewness_coeff",
        value=float(skew),
        extra={"skewness": float(skew), "n": len(x)},
    )


sskew = skewness_coeff


def cheatsheet() -> str:
    return "skewness_coeff({}) -> Skewness coefficient."
