# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Benford's law test for first-digit distribution."""

import numpy as np

from ._containers import DescriptiveResult


def benfords_law_test(data, cdf=None):
    """
    Test whether data follow Benford's law (first-digit law).

    Computes chi-squared goodness-of-fit between observed first-digit
    frequencies and the Benford distribution P(d) = log10(1 + 1/d).

    :param data: (n,) numeric data (positive values only).
    :return: DescriptiveResult with chi-squared statistic and p-value.

    References
    ----------
    Benford F (1938). The Law of Anomalous Numbers. Proc APS 78(4):551-572.
    Nigrini MJ (2012). Benford's Law. Wiley.
    """
    from scipy import stats as sp_stats

    arr = np.abs(np.asarray(data, dtype=np.float64).ravel())
    arr = arr[arr > 0]

    first_digits = np.array([int(str(x).lstrip("0").lstrip(".")[0]) for x in arr if x > 0])
    first_digits = first_digits[first_digits >= 1]

    observed = np.zeros(9)
    for d in first_digits:
        if 1 <= d <= 9:
            observed[d - 1] += 1

    n = observed.sum()
    expected_prop = np.log10(1 + 1.0 / np.arange(1, 10))
    expected = expected_prop * n

    chi2 = np.sum((observed - expected) ** 2 / expected)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=8)

    return DescriptiveResult(
        name="benfords_law_test",
        value=float(chi2),
        extra={
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "df": 8,
            "n": int(n),
            "observed_proportions": (observed / n).tolist() if n > 0 else [],
            "expected_proportions": expected_prop.tolist(),
            "conforms_to_benford": bool(p_value > 0.05),
        },
    )


def cheatsheet() -> str:
    return "benfords_law_test({}) -> Benford's law test for first-digit distribution."
