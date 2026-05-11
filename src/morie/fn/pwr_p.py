# morie.fn — function file (hadesllm/morie)
"""Power for two-proportion z-test."""

import math

import numpy as np
from statsmodels.stats.power import NormalIndPower


def power_prop_test(
    n: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two-sided",
) -> float:
    """
    Power for two-proportion z-test.

    Solves for one missing parameter among ``n``, ``p1``, ``p2``, or ``power``.
    Mirrors R's ``power.prop.test()``.

    :param n: Sample size per group.
    :param p1: Proportion in group 1.
    :param p2: Proportion in group 2.
    :param alpha: Type I error rate. Default 0.05.
    :param power: Desired power.
    :param alternative: ``"two-sided"`` or ``"one-sided"``. Default ``"two-sided"``.
    :return: The value of the missing parameter (n, or power).
    :raises ValueError: If p1 and p2 are both provided but either is out of [0, 1].

    Notes
    -----
    The NormalIndPower class operates on an arcsine-transformed effect size
    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)) (Cohen's h). This is the
    conventional approach for proportion tests.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
        Section 7. Effect size h.
    R Core Team (2024). power.prop.test {stats}. R documentation.
    """
    if p1 is not None and not 0 < p1 < 1:
        raise ValueError(f"p1 must be in (0, 1), got {p1}.")
    if p2 is not None and not 0 < p2 < 1:
        raise ValueError(f"p2 must be in (0, 1), got {p2}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    two_tailed = alternative == "two-sided"

    if p1 is not None and p2 is not None:
        # Cohen's h: effect size for proportions
        h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
        analysis = NormalIndPower()
        if n is None and power is not None:
            result = analysis.solve_power(
                effect_size=h,
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
            )
            return float(result)
        elif power is None and n is not None:
            result = analysis.solve_power(
                effect_size=h,
                alpha=float(alpha),
                nobs1=float(n),
                alternative="two-sided" if two_tailed else "larger",
            )
            return float(np.clip(result, 0.0, 1.0))
        else:
            raise ValueError("Provide exactly one of (n, power) when p1 and p2 are given.")
    else:
        raise ValueError("p1 and p2 must both be provided.")


pwr_p = power_prop_test


def cheatsheet() -> str:
    return "power_prop_test({}) -> Power for two-proportion z-test."
