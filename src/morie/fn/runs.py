# morie.fn -- function file (hadesllm/morie)
"""Wald-Wolfowitz runs test for randomness."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def runs_test(
    x: Union[list, np.ndarray],
) -> TestResult:
    r"""
    Wald-Wolfowitz runs test for randomness.

    Tests whether a binary sequence is randomly ordered by counting the
    number of *runs* (consecutive subsequences of identical values).

    Under H0 (random ordering), the number of runs *R* is approximately
    normal with:

    .. math::

        \\mu_R = \\frac{2 n_1 n_2}{n} + 1, \\quad
        \\sigma_R^2 = \\frac{2 n_1 n_2 (2 n_1 n_2 - n)}{n^2 (n - 1)}

    :param x: Binary array-like (values coerced to 0/1 via median split if
        not already binary).
    :return: TestResult with n_runs, z statistic, p_value.
    :raises ValueError: If x has fewer than 2 observations.

    References
    ----------
    Wald, A., & Wolfowitz, J. (1940). On a test whether two samples are from
        the same population. Annals of Mathematical Statistics, 11(2), 147-162.
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 2:
        raise ValueError("Runs test requires at least 2 observations.")

    # Coerce to binary via median split if needed
    unique = np.unique(arr)
    if len(unique) > 2:
        median = np.median(arr)
        binary = (arr > median).astype(int)
    elif len(unique) == 1:
        return TestResult(
            test_name="Runs",
            statistic=float("nan"),
            p_value=float("nan"),
            method="Wald-Wolfowitz runs test (constant input)",
            n=len(arr),
            extra={"n_runs": 1},
        )
    else:
        binary = (arr == unique[1]).astype(int)

    n = len(binary)
    n1 = int(np.sum(binary))
    n0 = n - n1

    if n0 == 0 or n1 == 0:
        return TestResult(
            test_name="Runs",
            statistic=float("nan"),
            p_value=float("nan"),
            method="Wald-Wolfowitz runs test (single category)",
            n=n,
            extra={"n_runs": 1, "n0": n0, "n1": n1},
        )

    # Count runs
    n_runs = 1 + int(np.sum(binary[1:] != binary[:-1]))

    # Expected value and variance
    mu = (2 * n0 * n1) / n + 1
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))

    if var <= 0:
        return TestResult(
            test_name="Runs",
            statistic=float(n_runs),
            p_value=float("nan"),
            method="Wald-Wolfowitz runs test (degenerate variance)",
            n=n,
            extra={"n_runs": n_runs, "n0": n0, "n1": n1},
        )

    z = (n_runs - mu) / np.sqrt(var)
    p_value = float(2 * stats.norm.sf(abs(z)))

    return TestResult(
        test_name="Runs",
        statistic=float(z),
        p_value=p_value,
        method="Wald-Wolfowitz runs test",
        n=n,
        extra={"n_runs": n_runs, "n0": n0, "n1": n1, "expected_runs": mu},
    )


runs = runs_test


def cheatsheet() -> str:
    return "runs_test({}) -> Wald-Wolfowitz runs test for randomness."
