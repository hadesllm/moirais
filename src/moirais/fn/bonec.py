# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Weibull failure analysis. 'I will crush you!' -- Bonecrusher"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def weibull_analysis(
    failure_times: np.ndarray,
    *,
    censored: np.ndarray | None = None,
    ci_level: float = 0.95,
) -> DescriptiveResult:
    """Weibull failure/reliability analysis via MLE.

    Fits a two-parameter Weibull distribution (shape k, scale lambda)
    to failure time data. Computes MTTF, reliability function, and
    hazard rate.

    Parameters
    ----------
    failure_times : array-like
        Time-to-failure observations.
    censored : array-like, optional
        Boolean mask (True = censored). Defaults to all uncensored.
    ci_level : float
        Confidence level for parameter CIs.

    Returns
    -------
    DescriptiveResult
        With ``value`` = dict(shape, scale, mttf) and ``extra``.
    """
    t = np.asarray(failure_times, dtype=float).ravel()
    t = t[t > 0]
    if len(t) < 3:
        raise ValueError("Need at least 3 positive failure times")

    if censored is not None:
        cens = np.asarray(censored, dtype=bool).ravel()
        if len(cens) != len(t):
            raise ValueError("censored must match failure_times length")
        uncensored = t[~cens]
    else:
        uncensored = t

    if len(uncensored) < 2:
        raise ValueError("Need at least 2 uncensored observations")

    shape, loc, scale = stats.weibull_min.fit(uncensored, floc=0)

    from scipy.special import gamma as gamma_fn

    mttf = scale * gamma_fn(1 + 1 / shape)

    eval_times = np.linspace(0.01, t.max() * 1.2, 100)
    reliability = np.exp(-((eval_times / scale) ** shape))
    hazard = (shape / scale) * (eval_times / scale) ** (shape - 1)

    n = len(uncensored)
    se_shape = shape / np.sqrt(n) * 1.1
    se_scale = scale / np.sqrt(n) * 1.1
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    shape_ci = (shape - z * se_shape, shape + z * se_shape)
    scale_ci = (scale - z * se_scale, scale + z * se_scale)

    return DescriptiveResult(
        name="weibull_analysis",
        value={"shape": float(shape), "scale": float(scale), "mttf": float(mttf)},
        extra={
            "n": len(t),
            "n_uncensored": len(uncensored),
            "shape_ci": shape_ci,
            "scale_ci": scale_ci,
            "eval_times": eval_times,
            "reliability": reliability,
            "hazard": hazard,
        },
    )


bonec = weibull_analysis


def cheatsheet() -> str:
    return "weibull_analysis({}) -> Weibull failure analysis. 'I will crush you!' -- Bonecrusher"
