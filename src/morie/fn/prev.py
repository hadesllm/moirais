# morie.fn — function file (hadesllm/morie)
"""Point prevalence with confidence interval."""

from __future__ import annotations

import math
from typing import Any


def point_prevalence(
    cases: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Compute point prevalence (proportion) with Wilson score CI.

    .. math::

        \\hat{p} = \\frac{x}{n}

    The Wilson score interval is preferred over the Wald interval for
    proportions, especially when p is near 0 or 1.

    :param cases: Number of cases (non-negative).
    :param total: Total population at the time point (positive).
    :param alpha: Significance level (default 0.05).
    :return: Dictionary with prevalence, se, ci_lower, ci_upper.
    :raises ValueError: If cases > total, cases < 0, or total <= 0.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22(158), 209--212.
    """
    from scipy import stats as _st

    if total <= 0:
        raise ValueError("total must be positive.")
    if cases < 0:
        raise ValueError("cases must be non-negative.")
    if cases > total:
        raise ValueError("cases cannot exceed total.")

    p = cases / total
    se = math.sqrt(p * (1 - p) / total) if total > 0 else 0.0

    # Wilson score interval
    z = _st.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom

    return {
        "prevalence": float(p),
        "se": float(se),
        "ci_lower": float(max(0.0, centre - margin)),
        "ci_upper": float(min(1.0, centre + margin)),
    }


prev = point_prevalence


def cheatsheet() -> str:
    return "point_prevalence({}) -> Point prevalence with confidence interval."
