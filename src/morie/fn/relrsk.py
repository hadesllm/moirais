# morie.fn — function file (hadesllm/morie)
"""Relative risk for 2x2 with R-style verbose result."""

from typing import Sequence, Union
import math
import numpy as np


def relrsk(table_2x2: Union[Sequence, np.ndarray]):
    """Relative risk for [[a,b],[c,d]]."""
    from ._richresult import RichResult
    t = np.asarray(table_2x2, dtype=float)
    if t.shape != (2, 2):
        raise ValueError(f"table must be 2x2, got {t.shape}.")
    p_exposed = t[0, 0] / max(t[0, 0] + t[0, 1], 1e-12)
    p_unexposed = t[1, 0] / max(t[1, 0] + t[1, 1], 1e-12)
    if p_unexposed == 0:
        raise ValueError("unexposed risk is zero - RR undefined.")
    rr = float(p_exposed / p_unexposed)
    log_rr = math.log(rr) if rr > 0 else float("-inf")
    n_exp = t[0, 0] + t[0, 1]; n_unexp = t[1, 0] + t[1, 1]
    if rr > 0 and t[0, 0] > 0 and t[1, 0] > 0:
        se = math.sqrt(1/t[0,0] - 1/n_exp + 1/t[1,0] - 1/n_unexp)
        ci_lo = math.exp(log_rr - 1.96 * se)
        ci_hi = math.exp(log_rr + 1.96 * se)
    else:
        se = ci_lo = ci_hi = float("nan")
    return RichResult(
        title="Relative risk (2x2)",
        summary_lines=[
            ("RR", rr),
            ("Risk in exposed", p_exposed),
            ("Risk in unexposed", p_unexposed),
            ("Risk difference", p_exposed - p_unexposed),
            ("95% CI for RR", f"[{ci_lo:.4g}, {ci_hi:.4g}]"),
            ("n exposed", int(n_exp)),
            ("n unexposed", int(n_unexp)),
        ],
        interpretation=(f"RR = {rr:.3g}; significant if 95% CI excludes 1.0. "
                        "RR > OR for rare events; RR ~ OR for very rare events."),
        payload={"value": rr, "statistic": rr,
                 "risk_exposed": p_exposed, "risk_unexposed": p_unexposed,
                 "ci_lo": ci_lo, "ci_hi": ci_hi},
    )
