# moirais.fn — function file (hadesllm/moirais)
"""Wilcoxon signed-rank test with R-style verbose result."""

from typing import Sequence, Union

import numpy as np
from scipy.stats import wilcoxon as _scipy_wilcoxon


def wilcoxn(x: Union[Sequence, np.ndarray],
            y: Union[Sequence, np.ndarray] = None,
            alternative: str = "two-sided"):
    """Wilcoxon signed-rank test (paired or one-sample).

    References
    ----------
    Wilcox (2017) ch.7.
    """
    from ._richresult import hypothesis_test_result
    a = np.asarray(x, dtype=float)
    if y is not None:
        b = np.asarray(y, dtype=float)
        diffs = a - b
        res = _scipy_wilcoxon(a, b, alternative=alternative)
    else:
        diffs = a
        res = _scipy_wilcoxon(a, alternative=alternative)
    return hypothesis_test_result(
        test_name="Wilcoxon signed-rank test (paired or one-sample)",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        extra_summary=[
            ("n observations", int(a.size)),
            ("Median of differences", float(np.median(diffs))),
            ("Alternative", alternative),
        ],
    )
