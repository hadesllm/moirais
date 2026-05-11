"""One-sample t-test against a specified null mean."""

from typing import Union

import numpy as np
import scipy.stats as stats


def one_sample_t_test(
    x: Union[list, np.ndarray],
    mu0: float = 0.0,
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    One-sample t-test against a specified null mean.

    Tests H0: mu = mu0 vs Ha depending on ``alternative``.

    :param x: Sample data (1-D array-like).
    :param mu0: Null hypothesis mean. Default 0.0.
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``t``, ``df``, ``p_value``, ``mean``, ``se``,
        ``ci_lower``, ``ci_upper``.
    :raises ValueError: If x has fewer than 2 observations.

    References
    ----------
    Student (1908). The probable error of a mean. Biometrika, 6(1), 1-25.
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 2:
        raise ValueError("x must have at least 2 observations.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    t_stat, p_val = stats.ttest_1samp(arr, popmean=mu0, alternative=alternative)
    n = len(arr)
    mean_val = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t(df=n - 1).ppf(0.975))
    return {
        "t": float(t_stat),
        "df": float(n - 1),
        "p_value": float(p_val),
        "mean": mean_val,
        "se": se,
        "ci_lower": mean_val - t_crit * se,
        "ci_upper": mean_val + t_crit * se,
        "method": "One-sample t-test",
    }


t1smp = one_sample_t_test


def cheatsheet() -> str:
    return "one_sample_t_test({}) -> One-sample t-test against a specified null mean."
