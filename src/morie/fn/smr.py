"""Standardized Mortality Ratio (SMR)."""

from __future__ import annotations

from typing import Any


def standardized_mortality_ratio(
    observed: int,
    expected: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the Standardized Mortality Ratio (SMR) with exact Poisson CI.

    .. math::

        \\text{SMR} = \\frac{O}{E}

    where O = observed deaths and E = expected deaths (from a reference
    population's age-specific rates applied to the study population).

    Uses exact Poisson confidence limits.

    :param observed: Observed number of deaths (non-negative integer).
    :param expected: Expected number of deaths (positive float).
    :param alpha: Significance level (default 0.05).
    :return: Dictionary with smr, ci_lower, ci_upper.
    :raises ValueError: If observed < 0 or expected <= 0.

    References
    ----------
    Breslow, N. E., & Day, N. E. (1987). *Statistical Methods in Cancer
    Research, Vol. II: The Design and Analysis of Cohort Studies*. IARC
    Scientific Publications No. 82, Ch. 2.
    """
    from scipy import stats as _st

    if observed < 0:
        raise ValueError("observed must be non-negative.")
    if expected <= 0:
        raise ValueError("expected must be positive.")

    smr_val = observed / expected

    # Exact Poisson CI on the count
    if observed == 0:
        ci_lo_count = 0.0
    else:
        ci_lo_count = _st.chi2.ppf(alpha / 2, 2 * observed) / 2
    ci_hi_count = _st.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / 2

    return {
        "smr": float(smr_val),
        "ci_lower": float(ci_lo_count / expected),
        "ci_upper": float(ci_hi_count / expected),
    }


smr = standardized_mortality_ratio


def cheatsheet() -> str:
    return "standardized_mortality_ratio({}) -> Standardized Mortality Ratio (SMR)."
