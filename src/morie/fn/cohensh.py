# morie.fn -- function file (hadesllm/morie)
"""Cohen's h (proportions) with R-style verbose result."""

import math


def cohensh(p1: float, p2: float):
    """Cohen's h for two proportions: 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))."""
    from ._richresult import RichResult
    for p in (p1, p2):
        if not 0 <= p <= 1:
            raise ValueError(f"proportions must be in [0, 1]; got {p}.")
    h = 2 * (math.asin(p1 ** 0.5) - math.asin(p2 ** 0.5))
    abs_h = abs(h)
    if abs_h < 0.2: bench = "negligible"
    elif abs_h < 0.5: bench = "small"
    elif abs_h < 0.8: bench = "medium"
    else: bench = "large"
    return RichResult(
        title="Cohen's h (effect size for two proportions)",
        summary_lines=[
            ("h", h),
            ("|h| benchmark", bench),
            ("p_1", p1), ("p_2", p2),
            ("Difference (p1 - p2)", p1 - p2),
        ],
        interpretation=(f"h = {h:+.3f}; |h| benchmarks: 0.2 small, 0.5 medium, "
                        "0.8 large (Cohen 1988)."),
        payload={"value": h, "statistic": h, "benchmark": bench},
    )
