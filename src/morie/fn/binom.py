# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Exact binomial test."""


import scipy.stats as stats

from ._containers import TestResult


def binomial_test(
    x: int,
    n: int,
    p0: float = 0.5,
    *,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Exact binomial test.

    Tests H0: the probability of success equals *p0*.

    :param x: Number of successes.
    :param n: Number of trials.
    :param p0: Null hypothesis proportion. Default 0.5.
    :param alternative: ``"two-sided"`` (default), ``"greater"``, or ``"less"``.
    :return: TestResult with p_value and observed proportion.
    :raises ValueError: If x > n, x < 0, n <= 0, or p0 not in [0, 1].

    References
    ----------
    Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.).
        Wiley. (Section 3.4.)
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if x < 0 or x > n:
        raise ValueError(f"x must be in [0, n], got x={x}, n={n}.")
    if not 0 <= p0 <= 1:
        raise ValueError(f"p0 must be in [0, 1], got {p0}.")

    result = stats.binomtest(x, n, p0, alternative=alternative)
    p_value = float(result.pvalue)
    p_hat = x / n

    return TestResult(
        test_name="Binomial",
        statistic=float(p_hat),
        p_value=p_value,
        method=f"Exact binomial test ({alternative})",
        n=n,
        extra={"x": x, "p0": p0, "p_hat": p_hat},
    )


binom = binomial_test


def cheatsheet() -> str:
    return "binomial_test({}) -> Exact binomial test."
