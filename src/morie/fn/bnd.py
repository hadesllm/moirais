# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Partial identification bounds (Manski bounds) under missing data."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def manski_bounds(
    outcome: Union[list, np.ndarray],
    treatment: Union[list, np.ndarray],
    missing: Union[list, np.ndarray],
    *,
    y_lower: float | None = None,
    y_upper: float | None = None,
) -> dict[str, Any]:
    """
    Compute Manski (1990) worst-case bounds on the ATE under missing data.

    For units with missing outcomes, the worst case is that their outcomes
    take the extreme values [y_lower, y_upper]. This yields an interval
    for E[Y(1)] and E[Y(0)] and hence for the ATE.

    If y_lower / y_upper are not provided, uses the observed min/max.

    :param outcome: Outcome variable (NaN or 0 for missing OK).
    :param treatment: Binary treatment (0/1).
    :param missing: Binary indicator (1 = outcome is missing, 0 = observed).
    :param y_lower: Assumed lower bound of outcome support.
    :param y_upper: Assumed upper bound of outcome support.
    :return: Dictionary with lower_bound, upper_bound, width.
    :raises ValueError: If all outcomes are missing.

    References
    ----------
    Manski, C. F. (1990). Nonparametric bounds on treatment effects.
    *American Economic Review: Papers and Proceedings*, 80(2), 319--323.

    Manski, C. F. (2003). *Partial Identification of Probability
    Distributions*. Springer.
    """
    Y = np.asarray(outcome, dtype=float)
    T = np.asarray(treatment, dtype=float).ravel()
    M = np.asarray(missing, dtype=float).ravel()
    n = len(Y)
    if len(T) != n or len(M) != n:
        raise ValueError("All arrays must have the same length.")

    observed = M == 0
    if not np.any(observed):
        raise ValueError("All outcomes are missing.")

    # Outcome bounds
    Y_obs = Y[observed]
    ylo = y_lower if y_lower is not None else float(np.min(Y_obs))
    yhi = y_upper if y_upper is not None else float(np.max(Y_obs))

    # For each arm, compute best/worst case mean
    def _bounds_arm(arm_val):
        arm_mask = arm_val == T
        obs_arm = arm_mask & observed
        mis_arm = arm_mask & (~observed)
        n_arm = int(np.sum(arm_mask))
        if n_arm == 0:
            return (float("nan"), float("nan"))
        n_obs = int(np.sum(obs_arm))
        n_mis = int(np.sum(mis_arm))
        sum_obs = float(np.sum(Y[obs_arm]))
        # Best case for this arm mean: observed sum + missing * yhi / n_arm (or ylo)
        mean_lo = (sum_obs + n_mis * ylo) / n_arm
        mean_hi = (sum_obs + n_mis * yhi) / n_arm
        return (mean_lo, mean_hi)

    e1_lo, e1_hi = _bounds_arm(1.0)
    e0_lo, e0_hi = _bounds_arm(0.0)

    # ATE bounds: ATE in [E[Y(1)]_lo - E[Y(0)]_hi, E[Y(1)]_hi - E[Y(0)]_lo]
    lb = e1_lo - e0_hi
    ub = e1_hi - e0_lo

    return {
        "lower_bound": float(lb),
        "upper_bound": float(ub),
        "width": float(ub - lb),
        "e1_bounds": (float(e1_lo), float(e1_hi)),
        "e0_bounds": (float(e0_lo), float(e0_hi)),
    }


bnd = manski_bounds


def cheatsheet() -> str:
    return "manski_bounds({}) -> Partial identification bounds (Manski bounds) under missing "
