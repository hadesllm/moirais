# morie.fn — function file (hadesllm/morie)
"""Kendall's tau-b with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import kendalltau


def kentau(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]):
    """Kendall's tau-b with tie correction."""
    from ._richresult import hypothesis_test_result
    a = np.asarray(x); b = np.asarray(y)
    res = kendalltau(a, b, variant="b")
    n = len(a)
    return hypothesis_test_result(
        test_name="Kendall's tau-b correlation",
        statistic=float(res.statistic), pvalue=float(res.pvalue),
        extra_summary=[
            ("n pairs", n),
            ("Strength", "negligible" if abs(res.statistic) < 0.1 else
                         "weak" if abs(res.statistic) < 0.3 else
                         "moderate" if abs(res.statistic) < 0.5 else "strong"),
            ("Direction", "positive" if res.statistic > 0 else "negative"),
        ],
    )
