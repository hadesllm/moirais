# morie.fn — function file (hadesllm/morie)
"""Odds ratio for 2x2 with R-style verbose result."""

from typing import Sequence, Union
import math
import numpy as np


def odds(table_2x2: Union[Sequence, np.ndarray], continuity: float = 0.0):
    """Odds ratio for [[a,b],[c,d]]: OR = (a*d) / (b*c)."""
    from ._richresult import RichResult
    t = np.asarray(table_2x2, dtype=float)
    if t.shape != (2, 2):
        raise ValueError(f"table must be 2x2, got {t.shape}.")
    if continuity:
        t = t + continuity
    if t[1, 0] == 0 or t[0, 1] == 0:
        raise ValueError("zero cell - pass continuity=0.5 for Haldane correction.")
    or_val = float((t[0, 0] * t[1, 1]) / (t[0, 1] * t[1, 0]))
    log_or = math.log(or_val)
    se_log_or = float(math.sqrt(sum(1.0 / x for x in t.flatten() if x > 0)))
    ci_lo = math.exp(log_or - 1.96 * se_log_or)
    ci_hi = math.exp(log_or + 1.96 * se_log_or)
    if or_val > 1: direction = "exposed group has HIGHER odds"
    elif or_val < 1: direction = "exposed group has LOWER odds"
    else: direction = "no association"
    return RichResult(
        title="Odds ratio (2x2)",
        summary_lines=[
            ("OR", or_val), ("log(OR)", log_or),
            ("SE(log OR)", se_log_or),
            ("95% CI for OR", f"[{ci_lo:.4g}, {ci_hi:.4g}]"),
            ("Direction", direction),
            ("Continuity correction", continuity if continuity else "none"),
            ("Cell a", float(t[0, 0])), ("Cell b", float(t[0, 1])),
            ("Cell c", float(t[1, 0])), ("Cell d", float(t[1, 1])),
        ],
        interpretation=(f"OR = {or_val:.3g} (95% CI [{ci_lo:.3g}, {ci_hi:.3g}]). "
                        f"Significant if CI excludes 1.0."),
        payload={"value": or_val, "statistic": or_val, "log_or": log_or,
                 "se_log_or": se_log_or, "ci_lo": ci_lo, "ci_hi": ci_hi},
    )
