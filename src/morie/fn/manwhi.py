# morie.fn — function file (hadesllm/morie)
"""Mann-Whitney U test with R-style verbose result."""

from typing import Sequence, Union

import numpy as np
from scipy.stats import mannwhitneyu


def manwhi(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray],
           alternative: str = "two-sided"):
    """Mann-Whitney U test for independent samples.

    Nonparametric alternative to the two-sample t-test. Tests stochastic
    dominance: P(X > Y) ≠ 0.5.

    References
    ----------
    Wilcox (2017) ch.7; Wooditch et al. (2021) ch.11.
    """
    from ._richresult import hypothesis_test_result
    a = np.asarray(x); b = np.asarray(y)
    res = mannwhitneyu(a, b, alternative=alternative)
    n1, n2 = len(a), len(b)
    warnings = []
    if n1 + n2 < 20:
        warnings.append(f"small total n ({n1+n2}); exact distribution may "
                        "be more reliable than the asymptotic Normal approx.")
    return hypothesis_test_result(
        test_name="Mann-Whitney U test (Wilcoxon rank-sum)",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        extra_summary=[
            ("n(x)", n1),
            ("n(y)", n2),
            ("Median(x)", float(np.median(a))),
            ("Median(y)", float(np.median(b))),
            ("Alternative", alternative),
        ],
        warnings=warnings,
    )
