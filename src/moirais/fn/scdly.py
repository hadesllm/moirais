# moirais.fn — function file (hadesllm/moirais)
"""Surveillance case reporting delay distribution."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def case_delay(
    delays: list[float] | np.ndarray,
    distribution: str = "gamma",
    confidence: float = 0.95,
) -> ESRes:
    """Estimate the reporting delay distribution for surveillance data.

    Reporting delay is the time between disease onset and case
    notification to the surveillance system.

    Parameters
    ----------
    delays : array-like of float
        Observed reporting delays (days).
    distribution : str, default 'gamma'
        Parametric family: 'gamma', 'lognormal', or 'exponential'.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Lawless, J. F. (1994). Adjustments for reporting delays and the
    prediction of occurred but not reported events. Canadian Journal
    of Statistics, 22(1), 15-31.
    """
    data = np.asarray(delays, dtype=float)
    data = data[data >= 0]
    if len(data) < 3:
        raise ValueError("Need at least 3 non-negative delays")

    if distribution == "gamma":
        shape, _, scale = stats.gamma.fit(data, floc=0)
        mean_val = shape * scale
        var_val = shape * scale**2
        params = {"shape": shape, "scale": scale}
    elif distribution == "lognormal":
        sigma, _, scale = stats.lognorm.fit(data[data > 0], floc=0)
        mean_val = np.exp(np.log(scale) + sigma**2 / 2)
        var_val = (np.exp(sigma**2) - 1) * np.exp(2 * np.log(scale) + sigma**2)
        params = {"sigma": sigma, "mu": np.log(scale)}
    elif distribution == "exponential":
        _, scale = stats.expon.fit(data, floc=0)
        mean_val = scale
        var_val = scale**2
        params = {"rate": 1.0 / scale}
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    se = np.sqrt(var_val / len(data))
    z = stats.norm.ppf((1 + confidence) / 2)
    pct_90 = float(np.percentile(data, 90))

    return ESRes(
        measure="case_delay",
        estimate=float(mean_val),
        se=float(se),
        ci_lower=float(mean_val - z * se),
        ci_upper=float(mean_val + z * se),
        n=len(data),
        extra={"distribution": distribution, "p90": pct_90, "variance": float(var_val), **params},
    )


scdly = case_delay


def cheatsheet() -> str:
    return "case_delay({}) -> Surveillance case reporting delay distribution."
