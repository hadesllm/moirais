# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bandwidth selection for regression discontinuity designs."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def rdd_bandwidth(
    running_var: Union[list, np.ndarray],
    outcome: Union[list, np.ndarray],
    *,
    cutoff: float = 0.0,
) -> dict[str, Any]:
    """
    Imbens-Kalyanaraman (IK) optimal bandwidth for sharp RDD.

    Simplified IK bandwidth selector:
    1. Estimate the variance of the outcome on each side of the cutoff.
    2. Estimate the second derivative of the conditional expectation
       using a global polynomial fit.
    3. Compute h_opt proportional to (variance / curvature)^(1/5) * n^(-1/5).

    Also returns h_half (h_opt/2) and h_double (h_opt*2) for sensitivity
    analysis.

    :param running_var: Running/assignment variable.
    :param outcome: Outcome variable.
    :param cutoff: RDD cutoff value.
    :return: Dictionary with h_opt, h_half, h_double, n_left, n_right.
    :raises ValueError: If fewer than 5 observations on either side.

    References
    ----------
    Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice
    for the regression discontinuity estimator. *Review of Economic
    Studies*, 79(3), 933--959.
    """
    rv = np.asarray(running_var, dtype=float)
    ov = np.asarray(outcome, dtype=float)
    if len(rv) != len(ov):
        raise ValueError("running_var and outcome must have the same length.")

    left_mask = rv < cutoff
    right_mask = rv >= cutoff
    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))
    if n_left < 5 or n_right < 5:
        raise ValueError("Need at least 5 observations on each side of cutoff.")

    n = len(rv)

    # Estimate variance on each side
    var_left = float(np.var(ov[left_mask], ddof=1))
    var_right = float(np.var(ov[right_mask], ddof=1))
    sigma2 = (var_left + var_right) / 2.0

    # Estimate second derivative via global quadratic fit
    # Center the running variable
    rc = rv - cutoff
    # Fit quadratic: y = a + b*rc + c*rc^2
    X_poly = np.column_stack([np.ones(n), rc, rc**2])
    beta = np.linalg.lstsq(X_poly, ov, rcond=None)[0]
    m2 = abs(float(beta[2]))  # second derivative estimate / 2
    curvature = max(2.0 * m2, 1e-10)  # f''(c)

    # IK bandwidth: h_opt = C_K * (sigma^2 / (curvature^2 * n))^(1/5)
    # C_K for triangular kernel ~ 3.4375
    C_K = 3.4375
    h_opt = C_K * (sigma2 / (curvature**2 * n)) ** 0.2

    # Clamp to data range
    data_range = float(np.max(rv) - np.min(rv))
    h_opt = min(h_opt, data_range / 2.0)

    return {
        "h_opt": float(h_opt),
        "h_half": float(h_opt / 2.0),
        "h_double": float(h_opt * 2.0),
        "n_left": n_left,
        "n_right": n_right,
    }


bwt = rdd_bandwidth


def cheatsheet() -> str:
    return "rdd_bandwidth({}) -> Bandwidth selection for regression discontinuity designs."
