# morie.fn — function file (hadesllm/morie)
"""Generation time distribution estimation."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def generation_time(
    intervals: list[float] | np.ndarray,
    distribution: str = "gamma",
    confidence: float = 0.95,
) -> ESRes:
    """Estimate the generation time distribution.

    Generation time is the interval between infection of a primary case
    and infection of a secondary case (vs serial interval which uses
    symptom onset).

    Parameters
    ----------
    intervals : array-like of float
        Observed generation times (days).
    distribution : str, default 'gamma'
        Parametric family: 'gamma' or 'lognormal'.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Svensson, A. (2007). A note on generation times in epidemic models.
    Mathematical Biosciences, 208(1), 300-311.

    Wallinga, J. & Lipsitch, M. (2007). How generation intervals shape
    the relationship between growth rates and reproductive numbers.
    Proceedings of the Royal Society B, 274(1609), 599-604.
    """
    data = np.asarray(intervals, dtype=float)
    data = data[data > 0]
    if len(data) < 3:
        raise ValueError("Need at least 3 positive generation times")

    if distribution == "gamma":
        shape, _, scale = stats.gamma.fit(data, floc=0)
        mean_val = shape * scale
        var_val = shape * scale**2
        params = {"shape": shape, "scale": scale}
    elif distribution == "lognormal":
        sigma, _, mu_scale = stats.lognorm.fit(data, floc=0)
        mean_val = np.exp(np.log(mu_scale) + sigma**2 / 2)
        var_val = (np.exp(sigma**2) - 1) * np.exp(2 * np.log(mu_scale) + sigma**2)
        params = {"sigma": sigma, "mu": np.log(mu_scale)}
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    se = np.sqrt(var_val / len(data))
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="generation_time",
        estimate=float(mean_val),
        se=float(se),
        ci_lower=float(mean_val - z * se),
        ci_upper=float(mean_val + z * se),
        n=len(data),
        extra={"distribution": distribution, "variance": float(var_val), **params},
    )


gntme = generation_time


def cheatsheet() -> str:
    return "generation_time({}) -> Generation time distribution estimation."
