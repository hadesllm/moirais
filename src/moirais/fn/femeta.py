# moirais.fn — function file (hadesllm/moirais)
"""Fixed-effects (inverse-variance weighted) meta-analytic pooling."""

import math
from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def fixed_effects_meta(estimates: Union[np.ndarray, list[float]], standard_errors: Union[np.ndarray, list[float]], confidence: float = 0.95, cdf=None) -> ESRes:
    """Fixed-effects (inverse-variance weighted) meta-analytic pooling.

    Parameters
    ----------
    estimates : array-like
        Effect-size estimates from k studies.
    standard_errors : array-like
        Standard errors of the estimates.
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    theta = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    w = 1 / se**2
    pooled = (w * theta).sum() / w.sum()
    pooled_se = math.sqrt(1 / w.sum())
    z = stats.norm.ppf((1 + confidence) / 2)
    q = float(((theta - pooled) ** 2 * w).sum())
    k = len(theta)
    p_q = 1 - stats.chi2.cdf(q, k - 1) if k > 1 else 1.0
    return ESRes(
        measure="Fixed-effects meta-analysis",
        estimate=float(pooled),
        ci_lower=float(pooled - z * pooled_se),
        ci_upper=float(pooled + z * pooled_se),
        se=float(pooled_se),
        n=k,
        extra={"Q": float(q), "Q_p_value": float(p_q)},
    )


fe_meta = fixed_effects_meta


def cheatsheet() -> str:
    return "fixed_effects_meta({}) -> Fixed-effects (inverse-variance weighted) meta-analytic pool"
