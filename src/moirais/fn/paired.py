# moirais.fn — function file (hadesllm/moirais)
"""Paired t-test with R-style verbose result."""

from typing import Sequence, Union

import numpy as np
from scipy.stats import ttest_rel


def paired(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]):
    """Paired t-test on matched x, y samples.

    Returns a ``RichResult``. Access ``.statistic``, ``.pvalue``, or
    print for a verbose multi-section summary including the mean
    difference, SD of differences, and assumption warnings.

    References
    ----------
    Wilcox (2017) ch.6; Wooditch et al. (2021) ch.11.
    """
    from ._richresult import hypothesis_test_result
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("x and y must be paired (same length); got "
                         f"x:{a.shape}, y:{b.shape}.")
    diff = a - b
    res = ttest_rel(a, b)

    warnings = []
    if a.size < 10:
        warnings.append(f"small sample (n={a.size}); paired-t assumes diffs "
                        "are approximately Normal — consider `wilcoxn` "
                        "(signed-rank) as a robust alternative.")

    return hypothesis_test_result(
        test_name="Paired t-test",
        statistic=float(res.statistic),
        df=float(a.size - 1),
        pvalue=float(res.pvalue),
        extra_summary=[
            ("n pairs", int(a.size)),
            ("Mean difference", float(diff.mean())),
            ("SD of differences", float(diff.std(ddof=1)) if diff.size > 1 else float("nan")),
        ],
        warnings=warnings,
    )
