# morie.fn -- function file (hadesllm/morie)
"""K-sample Anderson-Darling test. 'We are brave. -- Rose Tico'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def k_sample_anderson_darling(
    *samples: np.ndarray,
) -> TestResult:
    """K-sample Anderson-Darling test.

    Tests whether k independent samples come from a common
    (unspecified) continuous distribution.  Uses scipy's
    implementation of the Scholz & Stephens (1987) k-sample AD.

    :param samples: Two or more 1-D arrays of observations.
    :return: TestResult with AD statistic and approximate p-value.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples.")
    arrays = [np.asarray(s, dtype=float).ravel() for s in samples]
    result = stats.anderson_ksamp(arrays)
    stat = float(result.statistic)
    pval = float(result.pvalue) if hasattr(result, "pvalue") else float(result.significance_level)
    n_total = sum(len(a) for a in arrays)

    return TestResult(
        test_name="k-sample Anderson-Darling",
        statistic=stat,
        p_value=pval,
        method="Scholz-Stephens k-sample AD",
        n=n_total,
        extra={"k": len(arrays), "sample_sizes": [len(a) for a in arrays]},
    )


ksamp = k_sample_anderson_darling


def cheatsheet() -> str:
    return "k_sample_anderson_darling({}) -> K-sample Anderson-Darling test. 'We are brave. -- Rose Tico'"
