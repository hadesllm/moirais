# morie.fn -- function file (hadesllm/morie)
"""Welch's t-test (unequal variances) with R-style verbose result."""

from typing import Sequence, Union

import numpy as np
from scipy.stats import ttest_ind


def welcht(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]):
    """Welch's two-sample t-test (unequal variances assumed).

    Returns a ``RichResult`` -- print it for an R-style verbose summary,
    or access fields like ``.statistic``, ``.pvalue``, ``.df``.

    >>> r = welcht([1,2,3,4,5], [10,20,30,40,50])
    >>> r.statistic         # numeric access still works
    >>> print(r)            # multi-section summary

    References
    ----------
    Wilcox (2017) ch.6; Hedderich et al. (2023) ch.7.
    """
    from ._richresult import hypothesis_test_result
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    res = ttest_ind(a, b, equal_var=False)

    warnings = []
    if a.size < 5 or b.size < 5:
        warnings.append(
            "small sample size (n<5 in at least one group); the t-approximation "
            "may be unreliable. Consider Mann-Whitney (`manwhi`) instead."
        )
    if abs(a.mean() - b.mean()) < 1e-12:
        warnings.append("group means are essentially identical (Δ < 1e-12).")

    return hypothesis_test_result(
        test_name="Welch's two-sample t-test (unequal variances)",
        statistic=float(res.statistic),
        df=float(res.df),
        pvalue=float(res.pvalue),
        extra_summary=[
            (f"Mean(x), n={a.size}", float(a.mean())),
            (f"Mean(y), n={b.size}", float(b.mean())),
            ("Mean difference", float(a.mean() - b.mean())),
            ("SD(x)", float(a.std(ddof=1)) if a.size > 1 else float("nan")),
            ("SD(y)", float(b.std(ddof=1)) if b.size > 1 else float("nan")),
        ],
        warnings=warnings,
    )
