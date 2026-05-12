# morie.fn -- function file (hadesllm/morie)
"""Power for one-way ANOVA."""

import numpy as np
from statsmodels.stats.power import FTestAnovaPower


def power_anova(
    n: float | None = None,
    k: int | None = None,
    f: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> float:
    """
    Power for one-way ANOVA. Solves for one missing parameter.

    Mirrors R's ``power.anova.test()``.

    :param n: Number of observations per group.
    :param k: Number of groups.
    :param f: Cohen's f effect size (f = sigma_between / sigma_within).
        Conventional benchmarks: 0.10 small, 0.25 medium, 0.40 large.
    :param alpha: Type I error rate. Default 0.05.
    :param power: Desired power.
    :return: The value of the missing parameter.
    :raises ValueError: If more than one parameter is None or k < 2.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
        Section 8.
    R Core Team (2024). power.anova.test {stats}. R documentation.
    """
    none_count = sum(v is None for v in [n, k, f, power])
    if none_count != 1:
        raise ValueError("Exactly one of n, k, f, or power must be None.")
    if k is not None and k < 2:
        raise ValueError(f"k must be >= 2, got {k}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    analysis = FTestAnovaPower()

    if power is None:
        # nobs = total sample = n * k
        nobs = float(n * k)
        result = analysis.solve_power(
            effect_size=float(f),
            nobs=nobs,
            alpha=float(alpha),
            k_groups=int(k),
        )
        return float(np.clip(result, 0.0, 1.0))
    elif n is None:
        # Solve for n per group; FTestAnovaPower uses total nobs
        # We iterate: for a given k, find nobs then divide by k
        nobs = analysis.solve_power(
            effect_size=float(f),
            alpha=float(alpha),
            power=float(power),
            k_groups=int(k),
        )
        return float(nobs) / float(k)
    elif f is None:
        nobs = float(n * k)
        result = analysis.solve_power(
            nobs=nobs,
            alpha=float(alpha),
            power=float(power),
            k_groups=int(k),
        )
        return float(result)
    else:  # k is None -- iterate over k (integer) to find smallest k meeting power
        # FTestAnovaPower requires integer k; we solve by scanning
        for k_try in range(2, 200):
            nobs = float(n * k_try)
            try:
                pwr = analysis.solve_power(
                    effect_size=float(f),
                    nobs=nobs,
                    alpha=float(alpha),
                    k_groups=k_try,
                )
                if float(pwr) >= float(power):
                    return float(k_try)
            except Exception:
                continue
        raise ValueError("Could not find k in [2, 200) achieving the desired power.")


pwr_av = power_anova


def cheatsheet() -> str:
    return "power_anova({}) -> Power for one-way ANOVA."
