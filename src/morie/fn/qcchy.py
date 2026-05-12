# morie.fn -- function file (hadesllm/morie)
"""Cauchy distribution quantile function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qcchy(
    p: Union[float, np.ndarray], loc: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False
) -> Union[float, np.ndarray]:
    r"""
    Cauchy distribution quantile (inverse CDF).

    .. math::

        Q(p) = x_0 + \\gamma \\tan\\!\\left[\\pi(p - 1/2)\\right]

    Mirrors R's ``qcauchy(p, location, scale, lower.tail, log.p)``.

    :param p: Probability/probabilities in [0, 1].
    :param loc: Location parameter. Default 0.0.
    :param scale: Scale parameter (> 0). Default 1.0.
    :param lower_tail: If True (default), quantile of P(X <= x) = p.
    :param log_p: If True, *p* is given as log(p). Default False.
    :return: Quantile value(s).
    :raises ValueError: If scale <= 0.

    References
    ----------
    R Core Team (2024). qcauchy {stats}. R documentation.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - np.asarray(p)
    dist = stats.cauchy(loc=loc, scale=scale)
    return dist.ppf(p)


def cheatsheet() -> str:
    return "qcchy({}) -> Cauchy distribution quantile function."
