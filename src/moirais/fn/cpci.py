# moirais.fn — function file (hadesllm/moirais)
"""Clopper-Pearson exact binomial CI. 'We have hope. Rebellions are built on hope.' -- Jyn Erso"""

from __future__ import annotations

from scipy import stats as _st

from ._containers import DescriptiveResult


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> DescriptiveResult:
    """
    Clopper-Pearson exact binomial confidence interval.

    Uses the relationship between the binomial and Beta distributions:

    .. math::

        \\left[ B(\\alpha/2; x, n-x+1), \\;
               B(1-\\alpha/2; x+1, n-x) \\right]

    where :math:`B` is the Beta quantile function.

    :param successes: Number of successes.
    :type successes: int
    :param trials: Number of trials.
    :type trials: int
    :param alpha: Significance level. Default 0.05.
    :type alpha: float
    :return: DescriptiveResult with exact CI bounds.
    :rtype: DescriptiveResult
    :raises ValueError: If successes > trials or trials <= 0.

    References
    ----------
    Clopper C.J. & Pearson E.S. (1934). The use of confidence or
    fiducial limits illustrated in the case of the binomial.
    *Biometrika*, 26(4), 404-413.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}.")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}.")
    x, n = successes, trials
    lo = float(_st.beta.ppf(alpha / 2, x, n - x + 1)) if x > 0 else 0.0
    hi = float(_st.beta.ppf(1 - alpha / 2, x + 1, n - x)) if x < n else 1.0
    return DescriptiveResult(
        name="clopper_pearson",
        value=x / n,
        extra={"lower": lo, "upper": hi, "proportion": x / n, "alpha": alpha},
    )


cpci = clopper_pearson


def cheatsheet() -> str:
    return "clopper_pearson({}) -> Clopper-Pearson exact binomial CI. 'We have hope. Rebellions"
