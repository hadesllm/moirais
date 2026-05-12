# morie.fn -- function file (hadesllm/morie)
"""Shapiro-Wilk Normality test with R-style verbose result."""

from typing import Sequence, Union

import numpy as np
from scipy.stats import shapiro


def shapir(x: Union[Sequence, np.ndarray]):
    """Shapiro-Wilk test for Normality.

    References
    ----------
    Wilcox (2017) ch.5; Wooditch et al. (2021) ch.12.
    """
    from ._richresult import hypothesis_test_result
    a = np.asarray(x, dtype=float)
    if a.size < 3:
        raise ValueError(f"need at least 3 observations, got {a.size}.")
    res = shapiro(a)
    warnings = []
    if a.size > 5000:
        warnings.append(f"n={a.size} is very large; even tiny deviations "
                        "from Normal become 'significant'. Consider Q-Q plot "
                        "or `anddrl` (Anderson-Darling) as a complement.")
    return hypothesis_test_result(
        test_name="Shapiro-Wilk test for Normality",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        extra_summary=[
            ("n", int(a.size)),
            ("Sample mean", float(a.mean())),
            ("Sample SD", float(a.std(ddof=1))),
            ("Sample skewness", float(((a - a.mean()) ** 3).mean() / a.std() ** 3) if a.std() > 0 else 0.0),
        ],
        warnings=warnings,
    )
