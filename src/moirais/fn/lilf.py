# moirais.fn — function file (hadesllm/moirais)
"""Lilliefors test (KS with estimated parameters)."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def lilliefors_test(
    x: Union[list, np.ndarray],
) -> TestResult:
    """
    Lilliefors test for normality.

    A modification of the Kolmogorov-Smirnov test where the mean and
    standard deviation are estimated from the data rather than specified
    a priori. Because parameters are estimated, standard KS critical
    values are too liberal; the Lilliefors correction yields a more
    conservative test.

    Implementation: KS test against N(mean, sd) fitted from the data.
    P-value from ``scipy.stats.kstest`` (conservative for Lilliefors
    since scipy uses standard KS tables).

    :param x: Sample data (1-D array-like, n >= 4).
    :return: TestResult with D statistic and p_value.
    :raises ValueError: If x has fewer than 4 observations.

    References
    ----------
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality
        with mean and variance unknown. Journal of the American Statistical
        Association, 62(318), 399-402.
    """
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < 4:
        raise ValueError("Lilliefors test requires at least 4 observations.")

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))

    if sigma == 0:
        return TestResult(
            test_name="Lilliefors",
            statistic=0.0,
            p_value=1.0,
            method="Lilliefors test (zero variance)",
            n=n,
        )

    # KS test against fitted normal
    D, p_value = stats.kstest(arr, "norm", args=(mu, sigma))

    return TestResult(
        test_name="Lilliefors",
        statistic=float(D),
        p_value=float(p_value),
        method="Lilliefors test (KS with estimated params)",
        n=n,
        extra={"mu_hat": mu, "sigma_hat": sigma},
    )


lilf = lilliefors_test


def cheatsheet() -> str:
    return "lilliefors_test({}) -> Lilliefors test (KS with estimated parameters)."
