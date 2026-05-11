# morie.fn — function file (hadesllm/morie)
"""Cohen's d effect size with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def cohend(x1: Union[Sequence[float], np.ndarray],
           x2: Union[Sequence[float], np.ndarray]):
    """Cohen's d for two-sample mean difference.

    d = (xbar1 - xbar2) / s_pooled

    Returns RichResult — float(result) yields the d scalar; print()
    yields the multi-section guide. Conventional benchmarks: 0.2 small,
    0.5 medium, 0.8 large.

    References
    ----------
    Cohen (1988); Weisburd et al. (2022) ch.11 eq.11.1.
    """
    from ._richresult import RichResult
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError(f"each group needs at least 2 obs; got n1={a.size}, n2={b.size}.")
    n1, n2 = a.size, b.size
    v1 = a.var(ddof=1); v2 = b.var(ddof=1)
    s_pooled = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)))
    if s_pooled == 0:
        raise ValueError("pooled SD is zero - d is undefined.")
    d = float((a.mean() - b.mean()) / s_pooled)
    abs_d = abs(d)
    if abs_d < 0.2: bench = "negligible"
    elif abs_d < 0.5: bench = "small"
    elif abs_d < 0.8: bench = "medium"
    else: bench = "large"
    warnings = []
    if n1 < 30 or n2 < 30:
        warnings.append(f"small sample (n1={n1}, n2={n2}); consider hedgeg "
                        "(bias-corrected) for less optimistic effect estimate.")
    return RichResult(
        title="Cohen's d effect size",
        summary_lines=[
            ("d", d), ("|d| benchmark", bench),
            ("Mean(group 1)", float(a.mean())), ("Mean(group 2)", float(b.mean())),
            ("Mean diff", float(a.mean() - b.mean())),
            ("Pooled SD", s_pooled),
            ("n(group 1)", n1), ("n(group 2)", n2),
        ],
        warnings=warnings,
        interpretation=(f"d={d:.3f} -> {bench} effect (Cohen 1988 benchmarks: "
                        ".2/.5/.8 = small/medium/large)."),
        payload={"value": d, "statistic": d, "s_pooled": s_pooled, "benchmark": bench},
    )
