# morie.fn — function file (hadesllm/morie)
"""Calculate statistical power for an ANOVA F-test (interaction power)."""

import numpy as np
from statsmodels.stats.power import FTestAnovaPower


def calculate_interaction_power(sample_size: int, alpha: float = 0.05, effect_size: float = 0.2) -> float:
    r"""
    Calculate statistical power for an ANOVA F-test, used as an approximation
    for interaction-term power in the CPADS study design.

    The formula ``1 - exp(-(n * f) / 50)`` that appeared in earlier versions
    of this function is **not a valid power formula** -- it is a heuristic
    approximation with no grounding in probability theory and produces values
    that systematically diverge from the correct power at moderate sample
    sizes.  It has been replaced by the exact normal-approximation F-test
    power computation from :mod:`statsmodels.stats.power`.

    The implemented estimator solves:

    .. math::

        \\text{power} = 1 - F_{df_1, df_2, \\lambda}(F_{\\text{crit}})

    where :math:`F_{\\text{crit}}` is the critical value under the central
    F-distribution at level ``alpha``, :math:`\\lambda = n \\cdot f^2` is the
    non-centrality parameter, :math:`f` is Cohen's f effect size, :math:`df_1`
    defaults to 1 (single-df test), and :math:`df_2 = n - df_1 - 1`.

    :param sample_size: Total number of observations.
    :type sample_size: int
    :param alpha: Type I error probability limit (significance level), defaults to 0.05.
    :type alpha: float, optional
    :param effect_size: Cohen's *f* effect size for the interaction term,
        defaults to 0.2 (conventionally "small").
    :type effect_size: float, optional
    :return: The estimated statistical power (1 - beta), clipped to [0, 1].
    :rtype: float

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.

    Faul, F., Erdfelder, E., Lang, A.-G., & Buchner, A. (2007). G*Power 3: A
    flexible statistical power analysis program for the social, behavioral, and
    biomedical sciences. *Behavior Research Methods*, 39(2), 175-191.
    https://doi.org/10.3758/BF03193146
    """
    analysis = FTestAnovaPower()
    # solve_power returns a float; clip defensively to [0, 1].
    power = analysis.solve_power(
        effect_size=float(effect_size),
        nobs=float(sample_size),
        alpha=float(alpha),
    )
    return float(np.clip(power, 0.0, 1.0))


i_pwr = calculate_interaction_power


def cheatsheet() -> str:
    return "calculate_interaction_power({}) -> Calculate statistical power for an ANOVA F-test (interaction"
