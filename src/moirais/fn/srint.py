"""Serial interval estimation."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def serial_interval(
    intervals: list[float] | np.ndarray,
    distribution: str = "gamma",
    confidence: float = 0.95,
) -> ESRes:
    """Estimate the serial interval distribution from observed data.

    The serial interval is the time between symptom onset in a primary
    case and symptom onset in a secondary case.

    Parameters
    ----------
    intervals : array-like of float
        Observed serial intervals (days).
    distribution : str, default 'gamma'
        Parametric family to fit: 'gamma', 'lognormal', or 'weibull'.
    confidence : float, default 0.95
        Confidence level for parameter CIs.

    Returns
    -------
    ESRes

    References
    ----------
    Svensson, A. (2007). A note on generation times in epidemic models.
    Mathematical Biosciences, 208(1), 300-311.
    """
    data = np.asarray(intervals, dtype=float)
    data = data[data > 0]
    if len(data) < 3:
        raise ValueError("Need at least 3 positive serial intervals")

    if distribution == "gamma":
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        mean_val = shape * scale
        var_val = shape * scale**2
        params = {"shape": shape, "scale": scale}
    elif distribution == "lognormal":
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        mean_val = np.exp(np.log(scale) + shape**2 / 2)
        var_val = (np.exp(shape**2) - 1) * np.exp(2 * np.log(scale) + shape**2)
        params = {"sigma": shape, "mu": np.log(scale)}
    elif distribution == "weibull":
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)
        from scipy.special import gamma as gammafn
        mean_val = scale * gammafn(1 + 1 / shape)
        var_val = scale**2 * (gammafn(1 + 2 / shape) - gammafn(1 + 1 / shape) ** 2)
        params = {"shape": shape, "scale": scale}
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    se_mean = np.sqrt(var_val / len(data))
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="serial_interval",
        estimate=float(mean_val),
        se=float(se_mean),
        ci_lower=float(mean_val - z * se_mean),
        ci_upper=float(mean_val + z * se_mean),
        n=len(data),
        extra={"distribution": distribution, "variance": float(var_val), **params},
    )


srint = serial_interval


def cheatsheet() -> str:
    return "serial_interval({}) -> Serial interval distribution estimation."
