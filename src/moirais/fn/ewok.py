# moirais.fn — function file (hadesllm/moirais)
"""Glass's delta. 'Short help is still help.' -- Ewok proverb"""

from __future__ import annotations

import numpy as np

from ._containers import ESRes
from ._helpers import _arr, _bootstrap_ci


def glass_delta(x, y, *, ci: float = 0.95) -> ESRes:
    """Glass's delta: mean difference standardized by control group SD.

    Uses y (second group) as the reference/control group.

    Parameters
    ----------
    x : array-like
        Treatment / experimental group.
    y : array-like
        Control / reference group.
    ci : float
        Confidence level for bootstrap CI.

    Returns
    -------
    ESRes
    """
    x, y = _arr(x), _arr(y)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 obs per group")
    sd_ctrl = float(np.std(y, ddof=1))
    if sd_ctrl == 0:
        return ESRes(measure="Glass delta", estimate=float("nan"), n=len(x) + len(y))
    d = float((np.mean(x) - np.mean(y)) / sd_ctrl)
    se, ci_lo, ci_hi = _bootstrap_ci(lambda a, b: (a.mean() - b.mean()) / np.std(b, ddof=1), (x, y), confidence=ci)
    return ESRes(
        measure="Glass delta",
        estimate=d,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=len(x) + len(y),
    )


ewok = glass_delta


def cheatsheet() -> str:
    return "glass_delta({}) -> Glass's delta. 'Short help is still help.' -- Ewok proverb"
