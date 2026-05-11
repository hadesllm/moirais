"""Sign change test."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def sign_change_test(
    x,
    *,
    mu: float = 0.0,
    alternative: str = "two-sided",
) -> TestResult:
    """Sign test for the median.

    Counts observations above and below *mu* and uses the
    binomial distribution for inference.

    Parameters
    ----------
    x : array-like
        Observations.
    mu : float
        Hypothesised median (default 0).
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    from scipy.stats import binomtest

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    d = a - mu
    d = d[d != 0]
    n = len(d)
    if n < 1:
        raise ValueError("Need at least 1 non-zero difference.")

    n_pos = int(np.sum(d > 0))
    result = binomtest(n_pos, n, 0.5, alternative=alternative)

    return TestResult(
        test_name="Sign test",
        statistic=float(n_pos),
        p_value=float(result.pvalue),
        n=n,
        method=f"alternative={alternative}",
        extra={"n_positive": n_pos, "n_negative": n - n_pos},
    )


sgchg = sign_change_test


def cheatsheet() -> str:
    return "sign_change_test(x, mu=0) -> Sign test for the median."
