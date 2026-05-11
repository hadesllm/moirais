# morie.fn — function file (hadesllm/morie)
"""Spearman rank correlation with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import spearmanr


def spearm(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]):
    """Spearman's rank correlation rho."""
    from ._richresult import hypothesis_test_result
    a = np.asarray(x); b = np.asarray(y)
    res = spearmanr(a, b)
    return hypothesis_test_result(
        test_name="Spearman rank correlation",
        statistic=float(res.statistic), pvalue=float(res.pvalue),
        extra_summary=[
            ("n pairs", len(a)),
            ("Strength", "negligible" if abs(res.statistic) < 0.1 else
                         "weak" if abs(res.statistic) < 0.3 else
                         "moderate" if abs(res.statistic) < 0.5 else "strong"),
            ("Direction", "positive" if res.statistic > 0 else "negative"),
        ],
    )
