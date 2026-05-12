# morie.fn -- function file (hadesllm/morie)
"""Power analysis for t-tests."""

import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower


def power_t_test(
    n: float | None = None,
    delta: float | None = None,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two-sided",
    type: str = "two-sample",
) -> float:
    """
    Solve for any one missing parameter in a t-test power calculation.

    Exactly one of ``n``, ``delta``, or ``power`` must be None; the function
    solves for that parameter and returns it.

    Mirrors R's ``power.t.test()``.

    :param n: Sample size per group (two-sample) or total (one-sample).
    :param delta: Standardised effect size (|mean difference| / sd).
    :param sd: Standard deviation. Default 1.0.
    :param alpha: Type I error rate. Default 0.05.
    :param power: Desired power (1 - beta).
    :param alternative: ``"two-sided"`` or ``"one-sided"``. Default ``"two-sided"``.
    :param type: ``"two-sample"``, ``"one-sample"``, or ``"paired"``. Default ``"two-sample"``.
    :return: The value of the missing parameter.
    :raises ValueError: If exactly one parameter is not None, or invalid values provided.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    R Core Team (2024). power.t.test {stats}. R documentation.
    """
    none_count = sum(v is None for v in [n, delta, power])
    if none_count != 1:
        raise ValueError("Exactly one of n, delta, or power must be None.")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if power is not None and not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")

    if alternative == "two-sided":
        ratio = 1 if type in ("one-sample", "paired") else 1
        two_tailed = True
    elif alternative in ("one-sided", "greater", "less"):
        two_tailed = False
    else:
        raise ValueError(f"alternative must be 'two-sided' or 'one-sided', got {alternative!r}.")

    # Effective effect size (Cohen's d-like) = delta / sd
    effect = (delta / sd) if delta is not None else None

    if type == "two-sample":
        analysis = TTestIndPower()
        ratio_arg = 1.0  # equal group sizes
    else:
        analysis = TTestPower()
        ratio_arg = None

    if n is None:
        if type == "two-sample":
            result = analysis.solve_power(
                effect_size=float(effect),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
                ratio=ratio_arg,
            )
        else:
            result = analysis.solve_power(
                effect_size=float(effect),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
            )
        return float(result)
    elif delta is None:
        if type == "two-sample":
            result = analysis.solve_power(
                nobs1=float(n),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
                ratio=ratio_arg,
            )
        else:
            result = analysis.solve_power(
                nobs=float(n),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
            )
        return float(result) * float(sd)  # convert back to delta scale
    else:  # power is None
        if type == "two-sample":
            result = analysis.solve_power(
                effect_size=float(effect),
                nobs1=float(n),
                alpha=float(alpha),
                alternative="two-sided" if two_tailed else "larger",
                ratio=ratio_arg,
            )
        else:
            result = analysis.solve_power(
                effect_size=float(effect),
                nobs=float(n),
                alpha=float(alpha),
                alternative="two-sided" if two_tailed else "larger",
            )
        return float(np.clip(result, 0.0, 1.0))


pwr_t = power_t_test


def cheatsheet() -> str:
    return "power_t_test({}) -> Power analysis for t-tests."
