# morie.fn — function file (hadesllm/morie)
"""Hartigan's dip test for unimodality."""

from typing import Union

import numpy as np

from ._containers import TestResult


def dip_test(
    x: Union[list, np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int | None = None,
) -> TestResult:
    r"""
    Dip test for unimodality (Hartigan & Hartigan, 1985).

    Measures the maximum deviation of the empirical CDF from the
    best-fitting unimodal distribution. A large dip statistic suggests
    multimodality.

    The dip statistic D is defined as:

    .. math::

        D = \\inf_G \\sup_x |F_n(x) - G(x)|

    where :math:`G` ranges over all unimodal CDFs. This implementation
    uses a simplified algorithm that measures the maximum gap in the
    sorted data (normalised by range) as a proxy for multimodality,
    with bootstrap calibration under the null of unimodality.

    :param x: Sample data (1-D array-like, n >= 4).
    :param n_boot: Number of bootstrap replicates for p-value. Default 1000.
    :param seed: Random seed for reproducibility.
    :return: TestResult with dip statistic and bootstrap p_value.
    :raises ValueError: If x has fewer than 4 observations.

    References
    ----------
    Hartigan, J. A., & Hartigan, P. M. (1985). The dip test of unimodality.
        Annals of Statistics, 13(1), 70-84.
    """
    arr = np.sort(np.asarray(x, dtype=float))
    n = len(arr)
    if n < 4:
        raise ValueError("Dip test requires at least 4 observations.")

    dip = _compute_dip(arr, n)

    # Bootstrap p-value: generate from best-fit normal (unimodal null)
    # and compare dip statistics
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return TestResult(
            test_name="Dip",
            statistic=0.0,
            p_value=1.0,
            method="Hartigan's dip test (constant data)",
            n=n,
        )

    rng = np.random.default_rng(seed)
    n_extreme = 0
    for _ in range(n_boot):
        boot = np.sort(rng.normal(mu, sigma, size=n))
        if _compute_dip(boot, n) >= dip:
            n_extreme += 1

    p_value = (n_extreme + 1) / (n_boot + 1)

    return TestResult(
        test_name="Dip",
        statistic=float(dip),
        p_value=float(p_value),
        method="Hartigan's dip test (bootstrap)",
        n=n,
        extra={"n_boot": n_boot},
    )


def _compute_dip(sorted_x: np.ndarray, n: int) -> float:
    """Compute dip statistic from sorted data.

    Measures the largest normalised gap in sorted data. Under unimodality
    the data concentrates near the mode and gaps are small; under
    multimodality the antimode region produces a large gap.

    The statistic is: max(spacing_i / range) - 1/(n-1), clamped to >= 0.
    For unimodal data, this is typically near 0. For bimodal data with
    well-separated modes, this captures the gap.
    """
    if n < 2 or sorted_x[-1] == sorted_x[0]:
        return 0.0

    spacings = np.diff(sorted_x)
    total_range = sorted_x[-1] - sorted_x[0]
    norm_spacings = spacings / total_range
    expected_uniform = 1.0 / (n - 1)

    # Maximum excess spacing above uniform expectation
    max_excess = float(np.max(norm_spacings) - expected_uniform)
    return max(max_excess, 0.0)


dts = dip_test


def cheatsheet() -> str:
    return "dip_test({}) -> Hartigan's dip test for unimodality."
