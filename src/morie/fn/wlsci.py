"""Wilson score interval for binomial proportion. 'Waste no more time arguing what a good person should be. Be one. — Marcus Aurelius' -- General usage"""

from __future__ import annotations

from scipy import stats as _st

from ._containers import DescriptiveResult


def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> DescriptiveResult:
    r"""
    Wilson score confidence interval for a binomial proportion.

    .. math::

        \\tilde{p} \\pm \\frac{z}{1 + z^2/n}
        \\sqrt{\\frac{\\hat p(1 - \\hat p)}{n} + \\frac{z^2}{4n^2}}

    where :math:`\\tilde{p} = (\\hat p + z^2/2n) / (1 + z^2/n)`.

    :param successes: Number of successes.
    :type successes: int
    :param trials: Number of trials.
    :type trials: int
    :param alpha: Significance level. Default 0.05.
    :type alpha: float
    :return: DescriptiveResult with CI bounds and proportion estimate.
    :rtype: DescriptiveResult
    :raises ValueError: If successes > trials or trials <= 0.

    References
    ----------
    Wilson E.B. (1927). Probable inference, the law of succession, and
    statistical inference. *Journal of the American Statistical
    Association*, 22(158), 209-212.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}.")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}.")
    n = trials
    p_hat = successes / n
    z = _st.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z / denom * (p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return DescriptiveResult(
        name="wilson_ci",
        value=p_hat,
        extra={"lower": lo, "upper": hi, "proportion": p_hat, "alpha": alpha},
    )


wlsci = wilson_ci


def cheatsheet() -> str:
    return "wilson_ci({}) -> Wilson score interval for binomial proportion. 'May the Forc"
