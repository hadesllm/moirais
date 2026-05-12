# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Attack rate calculation."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def attack_rate(
    n_cases: int,
    n_exposed: int,
    *,
    alpha: float = 0.05,
) -> ESRes:
    """
    Compute the attack rate (proportion of exposed who became ill).

    .. math::

        AR = \\frac{\\text{cases}}{\\text{exposed}}

    Parameters
    ----------
    n_cases : int
        Number of cases (ill persons).
    n_exposed : int
        Number of persons exposed.
    alpha : float
        Significance level for CI.

    Returns
    -------
    ESRes

    References
    ----------
    Gregg, M. B. (2008). *Field Epidemiology*, 3rd ed. Oxford
    University Press, Ch. 4.
    """
    if n_cases < 0 or n_exposed <= 0:
        raise ValueError("n_cases >= 0 and n_exposed > 0 required.")
    if n_cases > n_exposed:
        raise ValueError("n_cases cannot exceed n_exposed.")

    ar = n_cases / n_exposed
    ci_lo, ci_hi = stats.binom.interval(1 - alpha, n_exposed, ar)
    ci_lo = ci_lo / n_exposed
    ci_hi = ci_hi / n_exposed
    se = np.sqrt(ar * (1 - ar) / n_exposed)

    return ESRes(
        measure="attack_rate",
        estimate=float(ar),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        se=float(se),
        n=n_exposed,
    )


attkr = attack_rate


def cheatsheet() -> str:
    return "attack_rate({}) -> Attack rate calculation."
