# morie.fn -- function file (hadesllm/morie)
"""Random-effects meta-analytic pooling (DerSimonian-Laird)."""

import math
from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def random_effects_meta(estimates: Union[np.ndarray, list[float]], standard_errors: Union[np.ndarray, list[float]], confidence: float = 0.95, method: str = "DL", cdf=None) -> ESRes:
    """Random-effects meta-analytic pooling.

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like
    confidence : float, default 0.95
    method : str, default "DL"
        Tau-squared estimator: "DL" (DerSimonian-Laird).

    Returns
    -------
    ESRes

    References
    ----------
    DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
    Controlled Clinical Trials, 7(3), 177-188.
    """
    theta = np.asarray(estimates, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    k = len(theta)
    w = 1 / se**2

    # Fixed-effects pooled
    theta_fe = (w * theta).sum() / w.sum()
    Q = float(((theta - theta_fe) ** 2 * w).sum())

    # DerSimonian-Laird tau-squared
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max((Q - (k - 1)) / c, 0.0) if c > 0 else 0.0

    # Random-effects weights
    w_re = 1 / (se**2 + tau2)
    pooled = (w_re * theta).sum() / w_re.sum()
    pooled_se = math.sqrt(1 / w_re.sum())
    z = stats.norm.ppf((1 + confidence) / 2)

    # I-squared
    i2 = max((Q - (k - 1)) / Q, 0.0) * 100 if Q > 0 else 0.0

    # Prediction interval
    pred_se = math.sqrt(pooled_se**2 + tau2)
    t_crit = stats.t.ppf((1 + confidence) / 2, max(k - 2, 1))
    pred_lo = pooled - t_crit * pred_se
    pred_hi = pooled + t_crit * pred_se

    return ESRes(
        measure="Random-effects meta-analysis (DL)",
        estimate=float(pooled),
        ci_lower=float(pooled - z * pooled_se),
        ci_upper=float(pooled + z * pooled_se),
        se=float(pooled_se),
        n=k,
        extra={
            "tau_squared": float(tau2),
            "tau": float(math.sqrt(tau2)),
            "I_squared": float(i2),
            "Q": float(Q),
            "Q_p_value": float(1 - stats.chi2.cdf(Q, k - 1)) if k > 1 else 1.0,
            "prediction_interval_lower": float(pred_lo),
            "prediction_interval_upper": float(pred_hi),
        },
    )


re_meta = random_effects_meta


def cheatsheet() -> str:
    return "random_effects_meta({}) -> Random-effects meta-analytic pooling (DerSimonian-Laird)."
