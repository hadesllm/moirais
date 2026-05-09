# moirais.fn — function file (hadesllm/moirais)
"""Glass's Δ effect size."""

from typing import Sequence, Union
import numpy as np
def glassd(treated, control) -> float:
    """Glass's Δ: (mean_t − mean_c) / sd_c.

    Standardizes by control SD only — appropriate when treatment
    can change variance, not just mean.
    """
    t = np.asarray(treated, dtype=float)
    c = np.asarray(control, dtype=float)
    sd_c = float(c.std(ddof=1))
    if sd_c == 0:
        raise ValueError("control SD is zero.")
    return (float(t.mean()) - float(c.mean())) / sd_c
