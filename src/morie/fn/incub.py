# morie.fn — function file (hadesllm/morie)
"""Incubation period distribution estimation."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def incubation_period(
    durations: list[float] | np.ndarray,
    distribution: str = "lognormal",
    confidence: float = 0.95,
) -> ESRes:
    """Estimate the incubation period distribution.

    The incubation period is the time from exposure to symptom onset.
    Lognormal is the default distribution following Lessler et al. (2009).

    Parameters
    ----------
    durations : array-like of float
        Observed incubation periods (days).
    distribution : str, default 'lognormal'
        Parametric family: 'lognormal', 'gamma', or 'weibull'.
    confidence : float, default 0.95
        Confidence level for mean CI.

    Returns
    -------
    ESRes

    References
    ----------
    Lessler, J. et al. (2009). Incubation periods of acute respiratory
    viral infections: a systematic review. Lancet Infectious Diseases,
    9(5), 291-300.
    """
    data = np.asarray(durations, dtype=float)
    data = data[data > 0]
    if len(data) < 3:
        raise ValueError("Need at least 3 positive durations")

    if distribution == "lognormal":
        sigma, _, scale = stats.lognorm.fit(data, floc=0)
        mu = np.log(scale)
        mean_val = np.exp(mu + sigma**2 / 2)
        median_val = np.exp(mu)
        var_val = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        pct_95 = float(stats.lognorm.ppf(0.95, sigma, scale=scale))
        params = {"mu": mu, "sigma": sigma, "median": float(median_val), "p95": pct_95}
    elif distribution == "gamma":
        shape, _, scale = stats.gamma.fit(data, floc=0)
        mean_val = shape * scale
        var_val = shape * scale**2
        median_val = float(stats.gamma.ppf(0.5, shape, scale=scale))
        pct_95 = float(stats.gamma.ppf(0.95, shape, scale=scale))
        params = {"shape": shape, "scale": scale, "median": median_val, "p95": pct_95}
    elif distribution == "weibull":
        shape, _, scale = stats.weibull_min.fit(data, floc=0)
        from scipy.special import gamma as gammafn
        mean_val = scale * gammafn(1 + 1 / shape)
        var_val = scale**2 * (gammafn(1 + 2 / shape) - gammafn(1 + 1 / shape) ** 2)
        params = {"shape": shape, "scale": scale}
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    se = np.sqrt(var_val / len(data))
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="incubation_period",
        estimate=float(mean_val),
        se=float(se),
        ci_lower=float(mean_val - z * se),
        ci_upper=float(mean_val + z * se),
        n=len(data),
        extra={"distribution": distribution, "variance": float(var_val), **params},
    )


incub = incubation_period


def cheatsheet() -> str:
    return "incubation_period({}) -> Incubation period distribution estimation."
