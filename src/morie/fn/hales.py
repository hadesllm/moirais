# morie.fn -- function file (hadesllm/morie)
"""Health-adjusted life expectancy (HALE)."""

from __future__ import annotations

from typing import Any

import numpy as np


def health_adjusted_life_expectancy(
    age_groups: np.ndarray,
    life_expectancy: np.ndarray,
    health_state_prevalence: np.ndarray,
    disability_weights: np.ndarray,
) -> dict[str, Any]:
    r"""Compute health-adjusted life expectancy (HALE).

    HALE adjusts life expectancy by subtracting years spent in less
    than full health, weighted by severity.

    .. math::

        HALE_x = LE_x - \\sum_s p_{x,s} \\cdot DW_s \\cdot LE_x

    For a single aggregate disability weight per age group:

    .. math::

        HALE_x = LE_x \\cdot (1 - DW_x^{\\text{agg}})

    Parameters
    ----------
    age_groups : array_like
        Starting ages of each group (e.g. [0, 1, 5, 10, ...]).
    life_expectancy : array_like
        Remaining life expectancy at each age group.
    health_state_prevalence : array_like
        Prevalence of ill-health in each age group (weighted average
        across all health states), values in [0, 1].
    disability_weights : array_like
        Average disability weight for each age group (weighted across
        health states), values in [0, 1].

    Returns
    -------
    dict
        Keys: 'hale' (array per age group), 'hale_at_birth', 'le_at_birth'.

    References
    ----------
    WHO (2020). WHO methods and data sources for life tables 1990-2019.
    Global Health Estimates Technical Paper WHO/DDI/DNA/GHE/2020.2.
    """
    ages = np.asarray(age_groups, dtype=float)
    le = np.asarray(life_expectancy, dtype=float)
    prev = np.asarray(health_state_prevalence, dtype=float)
    dw = np.asarray(disability_weights, dtype=float)

    if not (ages.shape == le.shape == prev.shape == dw.shape):
        raise ValueError("All arrays must have the same shape.")
    if np.any(prev < 0) or np.any(prev > 1):
        raise ValueError("Prevalence must be in [0, 1].")
    if np.any(dw < 0) or np.any(dw > 1):
        raise ValueError("Disability weights must be in [0, 1].")

    agg_dw = prev * dw
    hale = le * (1 - agg_dw)

    return {
        "hale": hale,
        "hale_at_birth": float(hale[0]) if len(hale) > 0 else np.nan,
        "le_at_birth": float(le[0]) if len(le) > 0 else np.nan,
    }


hales = health_adjusted_life_expectancy


def cheatsheet() -> str:
    return "health_adjusted_life_expectancy({}) -> HALE per age group."
