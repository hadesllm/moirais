# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Adaptive sample size re-estimation."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def adaptive_design(
    interim_effect: float,
    initial_n: int,
    *,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    max_increase: float = 2.0,
) -> ESRes:
    """
    Sample size re-estimation based on interim effect size.

    Recalculates required N using the observed interim effect and
    caps increase at max_increase * initial_n.

    Parameters
    ----------
    interim_effect : float
        Observed treatment effect at interim.
    initial_n : int
        Originally planned sample size per group.
    sd : float
        Assumed or observed standard deviation.
    alpha : float
        Significance level.
    power : float
        Target power.
    max_increase : float
        Maximum factor by which N can increase.

    Returns
    -------
    ESRes
        estimate = revised N per group.

    References
    ----------
    Cui, L., Hung, H. M., & Wang, S. J. (1999). Modification of
    sample size in group sequential clinical trials. *Biometrics*,
    55(3), 853-857.
    """
    if initial_n <= 0:
        raise ValueError("initial_n must be positive.")
    if abs(interim_effect) < 1e-12:
        return ESRes(
            measure="adaptive_design",
            estimate=float(int(np.ceil(initial_n * max_increase))),
            extra={"reason": "effect near zero, max increase applied"},
        )

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    revised_n = int(np.ceil(2 * ((z_a + z_b) * sd / interim_effect) ** 2))
    capped_n = min(revised_n, int(np.ceil(initial_n * max_increase)))
    capped_n = max(capped_n, initial_n)

    return ESRes(
        measure="adaptive_design",
        estimate=float(capped_n),
        extra={
            "revised_n": revised_n,
            "capped_n": capped_n,
            "initial_n": initial_n,
            "interim_effect": interim_effect,
        },
    )


adptv = adaptive_design


def cheatsheet() -> str:
    return "adaptive_design({}) -> Adaptive sample size re-estimation."
