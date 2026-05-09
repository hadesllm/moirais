"""Two-sample t-test (Welch or Student)."""

from typing import Union

import numpy as np
import scipy.stats as stats


def two_sample_t_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> dict:
    """
    Two-sample t-test (Welch or Student).

    Default is Welch's t-test (``equal_var=False``), which does not assume
    equal population variances and is generally safer (Ruxton, 2006).

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (1-D array-like).
    :param equal_var: If True use Student's t-test; if False (default) use Welch's.
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``t``, ``df``, ``p_value``, ``ci_diff_lower``,
        ``ci_diff_upper``, ``mean_diff``, ``method``.
    :raises ValueError: If x1 or x2 is empty or alternative is invalid.

    References
    ----------
    Welch, B. L. (1947). The generalization of Student's problem when several
        different population variances are involved. Biometrika, 34(1-2), 28-35.
    Ruxton, G. D. (2006). The unequal variance t-test is an underused alternative
        to Student's t-test and the Mann-Whitney U test. Behavioral Ecology, 17(4).
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 2:
        raise ValueError("x1 must have at least 2 observations.")
    if len(a2) < 2:
        raise ValueError("x2 must have at least 2 observations.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")

    t_stat, p_val = stats.ttest_ind(a1, a2, equal_var=equal_var, alternative=alternative)
    mean_diff = float(np.mean(a1) - np.mean(a2))
    # Degrees of freedom: Welch-Satterthwaite formula for unequal-var case
    if equal_var:
        df = float(len(a1) + len(a2) - 2)
    else:
        s1, s2 = np.var(a1, ddof=1), np.var(a2, ddof=1)
        n1, n2 = len(a1), len(a2)
        num = (s1 / n1 + s2 / n2) ** 2
        denom = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        df = float(num / denom) if denom > 0 else float(n1 + n2 - 2)
    # Two-sided CI for the mean difference
    se_diff = float(np.sqrt(np.var(a1, ddof=1) / len(a1) + np.var(a2, ddof=1) / len(a2)))
    t_crit = float(stats.t(df=df).ppf(0.975))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "t": float(t_stat),
        "df": df,
        "p_value": float(p_val),
        "mean_diff": mean_diff,
        "ci_diff_lower": ci_lower,
        "ci_diff_upper": ci_upper,
        "method": "Welch two-sample t-test" if not equal_var else "Student two-sample t-test",
    }


t2smp = two_sample_t_test


def cheatsheet() -> str:
    return "two_sample_t_test({}) -> Two-sample t-test (Welch or Student)."
