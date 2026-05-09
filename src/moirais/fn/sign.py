"""Sign test for paired data."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def sign_test(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray] | None = None,
    *,
    mu: float = 0.0,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Sign test for paired data or a single sample against a median.

    If *y* is provided, tests H0: median(x - y) = *mu*. Otherwise
    tests H0: median(x) = *mu*.

    The test statistic is the number of positive differences. Under H0
    this follows a Binomial(n_nonzero, 0.5) distribution.

    :param x: First sample (1-D array-like).
    :param y: Second sample for paired test (optional).
    :param mu: Hypothesised median difference. Default 0.
    :param alternative: ``"two-sided"`` (default), ``"greater"``, ``"less"``.
    :return: TestResult with n_pos, n_neg, p_value.
    :raises ValueError: If x and y have different lengths.

    References
    ----------
    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
        Inference (5th ed.). CRC Press. (Chapter 5.)
    """
    x_arr = np.asarray(x, dtype=float)
    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        if len(x_arr) != len(y_arr):
            raise ValueError(f"x and y must have the same length, got {len(x_arr)} and {len(y_arr)}.")
        diffs = x_arr - y_arr - mu
    else:
        diffs = x_arr - mu

    # Discard zeros (ties with null)
    nonzero = diffs[diffs != 0]
    n_nonzero = len(nonzero)

    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))

    if n_nonzero == 0:
        return TestResult(
            test_name="Sign",
            statistic=0.0,
            p_value=1.0,
            method="Sign test (no non-zero differences)",
            n=len(x_arr),
            extra={"n_pos": 0, "n_neg": 0, "n_ties": len(x_arr)},
        )

    # Two-sided binomial test
    result = stats.binomtest(n_pos, n_nonzero, 0.5, alternative=alternative)
    p_value = float(result.pvalue)

    return TestResult(
        test_name="Sign",
        statistic=float(n_pos),
        p_value=p_value,
        method=f"Sign test ({alternative})",
        n=len(x_arr),
        extra={"n_pos": n_pos, "n_neg": n_neg, "n_ties": len(x_arr) - n_nonzero},
    )


sign = sign_test


def cheatsheet() -> str:
    return "sign_test({}) -> Sign test for paired data."
