# morie.fn -- function file (hadesllm/morie)
"""Wald single-coefficient test with R-style verbose result."""

import math
from scipy.stats import norm, chi2


def wald(estimate: float, std_error: float,
         null_value: float = 0.0, test: str = "z"):
    """Wald test for H0: theta = theta0."""
    from ._richresult import hypothesis_test_result
    if std_error <= 0:
        raise ValueError(f"std_error must be positive, got {std_error}.")
    z = (estimate - null_value) / std_error
    if test == "z":
        p = 2 * (1 - norm.cdf(abs(z)))
        stat_label = "z"
        stat_val = z
    elif test == "chi2":
        w2 = z * z
        p = 1 - chi2.cdf(w2, df=1)
        stat_label = "Wald chi^2(1)"
        stat_val = w2
    else:
        raise ValueError(f"unknown test mode: {test!r}; use 'z' or 'chi2'.")
    return hypothesis_test_result(
        test_name=f"Wald test ({test})",
        statistic=float(stat_val), pvalue=float(p),
        extra_summary=[
            (stat_label, stat_val),
            ("Estimate", estimate),
            ("Standard error", std_error),
            ("Null value", null_value),
            ("|z|", abs(z)),
        ],
    )
