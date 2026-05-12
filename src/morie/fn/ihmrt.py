# morie.fn -- function file (hadesllm/morie)
"""Indigenous excess mortality ratio."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def indigenous_mortality(
    deaths_indigenous: int,
    deaths_general: int,
    pop_indigenous: int,
    pop_general: int,
    confidence: float = 0.95,
) -> ESRes:
    """Excess mortality ratio (Indigenous vs general population).

    Parameters
    ----------
    deaths_indigenous, deaths_general : int
    pop_indigenous, pop_general : int
    confidence : float

    Returns
    -------
    ESRes
    """
    if pop_indigenous <= 0 or pop_general <= 0:
        raise ValueError("Populations must be positive")

    rate_i = deaths_indigenous / pop_indigenous
    rate_g = deaths_general / pop_general
    ratio = rate_i / rate_g if rate_g > 0 else float("inf")

    log_ratio = np.log(ratio) if ratio > 0 and ratio != float("inf") else 0
    se_log = np.sqrt(1 / max(deaths_indigenous, 1) + 1 / max(deaths_general, 1))
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="excess_mortality_ratio",
        estimate=float(ratio),
        ci_lower=float(np.exp(log_ratio - z * se_log)),
        ci_upper=float(np.exp(log_ratio + z * se_log)),
        extra={"rate_indigenous": float(rate_i), "rate_general": float(rate_g)},
    )


ihmrt = indigenous_mortality


def cheatsheet() -> str:
    return "indigenous_mortality({}) -> Indigenous excess mortality ratio."
