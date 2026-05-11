# morie.fn — function file (hadesllm/morie)
"""Glass's delta effect size (control-group SD denominator)."""

import math
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._containers import ESRes
from ._helpers import _arr


def glass_delta(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    control: str = "y",
    confidence: float = 0.95,
) -> ESRes:
    """Glass's delta -- uses the control group SD as denominator.

    Parameters
    ----------
    x, y : array-like
    control : str, default "y"
        Which group is the control: ``"x"`` or ``"y"``.
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    x, y = _arr(x), _arr(y)
    ctrl = y if control == "y" else x
    sd_ctrl = ctrl.std(ddof=1)
    delta = (x.mean() - y.mean()) / sd_ctrl if sd_ctrl > 0 else 0.0
    n_ctrl = len(ctrl)
    se = math.sqrt(1 / len(x) + 1 / len(y) + delta**2 / (2 * (n_ctrl - 1)))
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Glass's delta",
        estimate=float(delta),
        ci_lower=float(delta - z * se),
        ci_upper=float(delta + z * se),
        se=float(se),
        n=len(x) + len(y),
    )


glassd = glass_delta


def cheatsheet() -> str:
    return "glass_delta({}) -> Glass's delta effect size (control-group SD denominator)."
