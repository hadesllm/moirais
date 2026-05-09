# moirais.fn — function file (hadesllm/moirais)
"""Kruskal-Wallis H-test with R-style verbose result."""

from typing import Sequence, Union

import numpy as np
from scipy.stats import kruskal


def kwallis(*groups: Union[Sequence, np.ndarray]):
    """Kruskal-Wallis H-test (nonparametric one-way ANOVA).

    References
    ----------
    Wilcox (2017) ch.10; Wooditch et al. (2021) ch.12.
    """
    from ._richresult import hypothesis_test_result
    if len(groups) < 2:
        raise ValueError(f"need at least 2 groups, got {len(groups)}.")
    arrs = [np.asarray(g) for g in groups]
    res = kruskal(*arrs)
    warnings = []
    if any(a.size < 5 for a in arrs):
        warnings.append("at least one group has n<5; the χ²-approximation "
                        "may be poor — consider exact permutation `permpv` for pairs.")
    return hypothesis_test_result(
        test_name="Kruskal-Wallis H-test (nonparametric ANOVA)",
        statistic=float(res.statistic),
        df=len(groups) - 1,
        pvalue=float(res.pvalue),
        extra_summary=[
            ("k (groups)", len(groups)),
            ("n total", sum(a.size for a in arrs)),
        ] + [(f"Median group {i+1}", float(np.median(a))) for i, a in enumerate(arrs)],
        warnings=warnings,
    )
