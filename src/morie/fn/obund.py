# morie.fn — function file (hadesllm/morie)
"""Manski partial identification bounds."""

from __future__ import annotations

from morie.fn._containers import ESRes


def otis_bounds(
    outcome_lower: float = 0.0,
    outcome_upper: float = 1.0,
    treatment_prob: float = 0.5,
    *,
    mean_treated: float = 0.0,
    mean_control: float = 0.0,
) -> ESRes:
    """Manski worst-case bounds for ATE under missing data.

    Parameters
    ----------
    outcome_lower, outcome_upper : float
        Bounds on the outcome variable.
    treatment_prob : float
        P(D=1).
    mean_treated, mean_control : float
        Observed conditional means.

    Returns
    -------
    ESRes
        ci_lower, ci_upper are the identification bounds.
    """
    lb = (treatment_prob * mean_treated + (1 - treatment_prob) * outcome_lower) - (
        treatment_prob * outcome_upper + (1 - treatment_prob) * mean_control
    )
    ub = (treatment_prob * mean_treated + (1 - treatment_prob) * outcome_upper) - (
        treatment_prob * outcome_lower + (1 - treatment_prob) * mean_control
    )
    mid = (lb + ub) / 2
    return ESRes(measure="otis_bounds", estimate=mid, ci_lower=lb, ci_upper=ub)


obund = otis_bounds


def cheatsheet() -> str:
    return "otis_bounds({}) -> Manski partial identification bounds."
