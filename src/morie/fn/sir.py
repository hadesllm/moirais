"""Standardized Incidence Ratio (SIR)."""

from __future__ import annotations

from typing import Any


def standardized_incidence_ratio(
    observed: int,
    expected: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the Standardized Incidence Ratio (SIR) with exact Poisson CI.

    .. math::

        \\text{SIR} = \\frac{O}{E}

    Identical in form to the SMR but applied to incident cases rather
    than deaths. Uses exact Poisson confidence limits.

    :param observed: Observed number of incident cases.
    :param expected: Expected number based on reference rates.
    :param alpha: Significance level (default 0.05).
    :return: Dictionary with sir, ci_lower, ci_upper.
    :raises ValueError: If observed < 0 or expected <= 0.

    References
    ----------
    Breslow, N. E., & Day, N. E. (1987). *Statistical Methods in Cancer
    Research, Vol. II*. IARC, Ch. 2.
    """
    from scipy import stats as _st

    if observed < 0:
        raise ValueError("observed must be non-negative.")
    if expected <= 0:
        raise ValueError("expected must be positive.")

    sir_val = observed / expected

    if observed == 0:
        ci_lo_count = 0.0
    else:
        ci_lo_count = _st.chi2.ppf(alpha / 2, 2 * observed) / 2
    ci_hi_count = _st.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / 2

    return {
        "sir": float(sir_val),
        "ci_lower": float(ci_lo_count / expected),
        "ci_upper": float(ci_hi_count / expected),
    }


sir = standardized_incidence_ratio


def cheatsheet() -> str:
    return "standardized_incidence_ratio({}) -> Standardized Incidence Ratio (SIR)."
