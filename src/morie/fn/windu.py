"""Wald test. 'This party is over.' -- Mace Windu"""

from __future__ import annotations

from scipy import stats

from ._containers import TestResult


def wald_test(estimate: float, se: float, null: float = 0.0) -> TestResult:
    """Wald test: (estimate - null) / se ~ N(0,1).

    :param estimate: Point estimate.
    :param se: Standard error (must be positive).
    :param null: Null hypothesis value (default 0).
    :return: TestResult with z-statistic and two-sided p-value.
    :raises ValueError: If se <= 0.
    """
    if se <= 0:
        raise ValueError("Standard error must be positive")
    z = (estimate - null) / se
    p = 2 * stats.norm.sf(abs(z))
    return TestResult(
        test_name="Wald",
        statistic=float(z),
        p_value=float(p),
        df=1,
        method="Wald z-test",
    )


windu = wald_test


def cheatsheet() -> str:
    return "wald_test({}) -> Wald test. 'This party is over.' -- Mace Windu"
