# morie.fn -- function file (hadesllm/morie)
"""Generation time distribution estimation."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def generation_time(
    generation_times: np.ndarray | list,
    *,
    distribution: str = "gamma",
) -> DescriptiveResult:
    """
    Estimate generation time distribution from contact tracing data.

    The generation time is the interval between infection of a primary
    case and infection of the secondary case they caused.

    Parameters
    ----------
    generation_times : array-like
        Observed generation times (positive values).
    distribution : str
        Parametric family to fit: 'gamma', 'lognormal', 'weibull'.

    Returns
    -------
    DescriptiveResult
        extra has 'mean', 'sd', 'params', 'aic'.

    References
    ----------
    Wallinga, J., & Lipsitch, M. (2007). How generation intervals
    shape the relationship between growth rates and reproductive
    numbers. *Proc R Soc B*, 274(1609), 599-604.
    """
    gt = np.asarray(generation_times, dtype=float)
    gt = gt[gt > 0]
    if len(gt) < 3:
        raise ValueError("Need at least 3 positive generation times.")

    mean_gt = float(np.mean(gt))
    sd_gt = float(np.std(gt, ddof=1))

    if distribution == "gamma":
        params = stats.gamma.fit(gt, floc=0)
        ll = np.sum(stats.gamma.logpdf(gt, *params))
        k_params = 2
    elif distribution == "lognormal":
        params = stats.lognorm.fit(gt, floc=0)
        ll = np.sum(stats.lognorm.logpdf(gt, *params))
        k_params = 2
    elif distribution == "weibull":
        params = stats.weibull_min.fit(gt, floc=0)
        ll = np.sum(stats.weibull_min.logpdf(gt, *params))
        k_params = 2
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    aic = 2 * k_params - 2 * ll

    return DescriptiveResult(
        name="generation_time",
        value=mean_gt,
        extra={
            "mean": mean_gt,
            "sd": sd_gt,
            "distribution": distribution,
            "params": params,
            "aic": float(aic),
            "n": len(gt),
        },
    )


gtime = generation_time


def cheatsheet() -> str:
    return "generation_time({}) -> Generation time distribution estimation."
