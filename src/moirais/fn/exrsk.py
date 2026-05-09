# moirais.fn — function file (hadesllm/moirais)
"""Excess absolute risk."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def excess_risk(
    rate_exposed: float,
    rate_unexposed: float,
    *,
    n_exposed: int | None = None,
    n_unexposed: int | None = None,
    alpha: float = 0.05,
) -> ESRes:
    """
    Compute excess absolute risk (risk difference).

    .. math::

        EAR = R_1 - R_0

    Parameters
    ----------
    rate_exposed : float
        Rate or risk in exposed group.
    rate_unexposed : float
        Rate or risk in unexposed group.
    n_exposed, n_unexposed : int, optional
        Sample sizes (for CI).
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern
    Epidemiology*, 3rd ed. Lippincott Williams & Wilkins, Ch. 4.
    """
    ear = rate_exposed - rate_unexposed
    ci_lo, ci_hi, se = None, None, None

    if n_exposed is not None and n_unexposed is not None:
        se1 = np.sqrt(rate_exposed * (1 - rate_exposed) / n_exposed) if n_exposed > 0 else 0
        se0 = np.sqrt(rate_unexposed * (1 - rate_unexposed) / n_unexposed) if n_unexposed > 0 else 0
        se = np.sqrt(se1**2 + se0**2)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lo = ear - z * se
        ci_hi = ear + z * se

    return ESRes(
        measure="excess_risk",
        estimate=float(ear),
        ci_lower=float(ci_lo) if ci_lo is not None else None,
        ci_upper=float(ci_hi) if ci_hi is not None else None,
        se=float(se) if se is not None else None,
    )


exrsk = excess_risk


def cheatsheet() -> str:
    return "excess_risk({}) -> Excess absolute risk."
