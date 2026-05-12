# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Agresti-Coull interval for binomial proportion. 'Rebellions are built on hope.' -- Cassian Andor"""

from __future__ import annotations

from scipy import stats as _st

from ._containers import DescriptiveResult


def agresti_coull(successes: int, trials: int, alpha: float = 0.05) -> DescriptiveResult:
    r"""
    Agresti-Coull confidence interval for a binomial proportion.

    Adds :math:`z^2/2` pseudo-successes and :math:`z^2/2`
    pseudo-failures before computing a Wald interval:

    .. math::

        \\tilde{n} = n + z^2, \\quad
        \\tilde{p} = \\frac{x + z^2/2}{\\tilde{n}}, \\quad
        \\tilde{p} \\pm z \\sqrt{\\frac{\\tilde{p}(1-\\tilde{p})}{\\tilde{n}}}

    :param successes: Number of successes.
    :type successes: int
    :param trials: Number of trials.
    :type trials: int
    :param alpha: Significance level. Default 0.05.
    :type alpha: float
    :return: DescriptiveResult with CI bounds.
    :rtype: DescriptiveResult
    :raises ValueError: If successes > trials or trials <= 0.

    References
    ----------
    Agresti A. & Coull B.A. (1998). Approximate is better than exact for
    interval estimation of binomial proportions. *The American
    Statistician*, 52(2), 119-126.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}.")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}.")
    z = _st.norm.ppf(1 - alpha / 2)
    n_tilde = trials + z**2
    p_tilde = (successes + z**2 / 2) / n_tilde
    margin = z * (p_tilde * (1 - p_tilde) / n_tilde) ** 0.5
    lo = max(0.0, p_tilde - margin)
    hi = min(1.0, p_tilde + margin)
    return DescriptiveResult(
        name="agresti_coull",
        value=p_tilde,
        extra={
            "lower": lo,
            "upper": hi,
            "adjusted_proportion": p_tilde,
            "raw_proportion": successes / trials,
            "alpha": alpha,
        },
    )


agci = agresti_coull


def cheatsheet() -> str:
    return "agresti_coull({}) -> Agresti-Coull interval for binomial proportion. 'Rebellions "
