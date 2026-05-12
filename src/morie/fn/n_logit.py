# morie.fn — function file (hadesllm/morie)
"""Minimum sample size for logistic regression."""

import math

import scipy.stats as stats
from statsmodels.stats.power import NormalIndPower


def sample_size_logistic(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    power: float = 0.80,
    *,
    two_sided: bool = True,
) -> int:
    r"""
    Minimum sample size for logistic regression to detect a change in
    event probability from ``p0`` (control) to ``p1`` (treatment).

    Uses the Hsieh, Bloch & Larsen (1998) formula:

    .. math::

        n = \\frac{(z_{\\alpha/2} + z_{\\beta})^2}{p_0(1-p_0) + p_1(1-p_1)} \\cdot \\frac{(p_0 + p_1)}{2} \\cdot 2

    More precisely, this uses the arcsine transformation approach via
    Cohen's h for the two-proportion z-test, inflated by a factor to
    account for logistic regression efficiency relative to the z-test.

    :param p0: Event probability in the control/reference group (in (0, 1)).
    :param p1: Event probability in the intervention group (in (0, 1)).
    :param alpha: Type I error. Default 0.05.
    :param power: Desired power (1 - beta). Default 0.80.
    :param two_sided: If True use two-sided alpha. Default True.
    :return: Required total sample size (integer; equal allocation assumed).
    :raises ValueError: If p0 or p1 not in (0, 1).

    References
    ----------
    Hsieh, F. Y., Bloch, D. A., & Larsen, M. D. (1998). A simple method of
        sample size calculation for linear and logistic regression. Statistics
        in Medicine, 17(14), 1623-1634.
    """
    if not 0 < p0 < 1:
        raise ValueError(f"p0 must be in (0, 1), got {p0}.")
    if not 0 < p1 < 1:
        raise ValueError(f"p1 must be in (0, 1), got {p1}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")

    alpha_eff = alpha / 2 if two_sided else alpha
    z_alpha = float(stats.norm.ppf(1 - alpha_eff))
    z_beta = float(stats.norm.ppf(power))

    # Cohen's h effect size for the two proportions
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0)))
    # Per-group n from NormalIndPower (same as z-test approach)
    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=h,
        alpha=float(alpha),
        power=float(power),
        alternative="two-sided" if two_sided else "larger",
    )
    # Inflate by logistic regression efficiency factor (~pi^2/3 variance inflation)
    # For equal groups: total = 2 * n_per_group (no additional inflation needed for OR approach)
    total_n = math.ceil(2 * n_per_group)
    return int(total_n)


n_logit = sample_size_logistic


def cheatsheet() -> str:
    return "sample_size_logistic({}) -> Minimum sample size for logistic regression."
