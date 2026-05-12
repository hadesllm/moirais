# morie.fn -- function file (hadesllm/morie)
"""Pearson r as an effect size with Fisher z confidence interval."""

import math
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._containers import ESRes
from ._helpers import _arr


def r_effect_size(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> ESRes:
    """Pearson r as an effect size with Fisher z CI.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    x, y = _arr(x), _arr(y)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    r, _ = stats.pearsonr(x, y)
    z_r = np.arctanh(r)
    se_z = 1 / math.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Pearson r",
        estimate=float(r),
        ci_lower=float(np.tanh(z_r - z_crit * se_z)),
        ci_upper=float(np.tanh(z_r + z_crit * se_z)),
        se=float(se_z),
        n=n,
    )


r_es = r_effect_size


def cheatsheet() -> str:
    return "r_effect_size({}) -> Pearson r as an effect size with Fisher z confidence interva"
