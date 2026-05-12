"""Kurtosis coefficient (excess)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Time discovers truth. -- Seneca"


def kurtosis_coeff(x, **kwargs) -> DescriptiveResult:
    r"""Compute the excess kurtosis of signal *x*.

    .. math::

        \\gamma_2 = \\frac{\\mu_4}{\\sigma^4} - 3

    where :math:`\\mu_4` is the fourth central moment.

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
        kurt = 0.0
    else:
        mu4 = float(np.mean((x - mu) ** 4))
        kurt = mu4 / (sigma**4) - 3.0
    return DescriptiveResult(
        name="kurtosis_coeff",
        value=float(kurt),
        extra={"excess_kurtosis": float(kurt), "n": len(x)},
    )


skurt = kurtosis_coeff


def cheatsheet() -> str:
    return "kurtosis_coeff({}) -> Kurtosis coefficient (excess)."
