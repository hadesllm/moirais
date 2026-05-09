# moirais.fn — function file (hadesllm/moirais)
"""Kolmogorov-Smirnov test with R-style verbose result."""

from typing import Sequence, Union, Callable
import numpy as np
from scipy.stats import ks_1samp, ks_2samp


def ksonebs(x: Union[Sequence, np.ndarray],
            cdf_or_y: Union[Callable, str, Sequence, np.ndarray] = "norm",
            alternative: str = "two-sided"):
    """Kolmogorov-Smirnov goodness-of-fit / two-sample test."""
    from ._richresult import hypothesis_test_result
    from scipy import stats as _ss
    a = np.asarray(x, dtype=float)
    if isinstance(cdf_or_y, str):
        dist = getattr(_ss, cdf_or_y, None)
        if dist is None:
            raise ValueError(f"unknown distribution: {cdf_or_y}; valid: 'norm','uniform','t','expon'.")
        target_name = cdf_or_y
        cdf_or_y = dist.cdf
        kind = "one-sample"
    elif callable(cdf_or_y):
        target_name = getattr(cdf_or_y, "__name__", "custom CDF")
        kind = "one-sample"
    else:
        target_name = "two-sample"
        kind = "two-sample"
    if kind == "one-sample":
        res = ks_1samp(a, cdf_or_y, alternative=alternative)
        n2_extra = []
    else:
        b = np.asarray(cdf_or_y, dtype=float)
        res = ks_2samp(a, b, alternative=alternative)
        n2_extra = [("n(y)", int(b.size)), ("Median(y)", float(np.median(b)))]
    return hypothesis_test_result(
        test_name=f"Kolmogorov-Smirnov test ({kind})",
        statistic=float(res.statistic), pvalue=float(res.pvalue),
        extra_summary=[("Test mode", kind), ("Target", target_name),
                       ("n(x)", int(a.size)), ("Median(x)", float(np.median(a)))]
                       + n2_extra + [("Alternative", alternative)],
    )
