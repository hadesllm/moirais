"""Sample size for comparing two means."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def sample_size_means(
    delta: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> ESRes:
    """
    Sample size for a two-sample t-test comparing means.

    .. math::

        n_1 = \\frac{(z_{\\alpha/2} + z_{\\beta})^2 \\sigma^2 (1 + 1/r)}{\\delta^2}

    Parameters
    ----------
    delta : float
        Minimum detectable difference.
    sd : float
        Common standard deviation.
    alpha : float
        Two-sided significance level.
    power : float
        Desired power (1 - beta).
    ratio : float
        Allocation ratio n2/n1.

    Returns
    -------
    ESRes
        estimate = n per group (group 1).

    References
    ----------
    Chow, S. C., Shao, J., & Wang, H. (2008). *Sample Size
    Calculations in Clinical Research*, 2nd ed. Chapman & Hall, Ch. 3.
    """
    if delta <= 0 or sd <= 0:
        raise ValueError("delta and sd must be positive.")
    if not (0 < alpha < 1) or not (0 < power < 1):
        raise ValueError("alpha and power must be in (0, 1).")

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n1 = (z_a + z_b) ** 2 * sd**2 * (1 + 1 / ratio) / delta**2
    n1 = int(np.ceil(n1))
    n2 = int(np.ceil(n1 * ratio))

    return ESRes(
        measure="sample_size_means",
        estimate=float(n1),
        extra={"n1": n1, "n2": n2, "total": n1 + n2, "delta": delta, "sd": sd},
    )


ssiz = sample_size_means


def cheatsheet() -> str:
    return "sample_size_means({}) -> Sample size for comparing two means."
