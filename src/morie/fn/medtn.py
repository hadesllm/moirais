# morie.fn — function file (hadesllm/morie)
"""Causal mediation analysis (Baron-Kenny approach)."""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np


def causal_mediation(X: Union[list, np.ndarray], M: Union[list, np.ndarray], Y: Union[list, np.ndarray], cdf=None) -> dict[str, Any]:
    """
    Causal mediation analysis using the Baron-Kenny (1986) approach.

    Fits three OLS regressions:
      1. Y = c * X + e1           (total effect)
      2. M = a * X + e2           (X -> M path)
      3. Y = c' * X + b * M + e3 (direct + mediated)

    Indirect effect = a * b.  Proportion mediated = (a * b) / c.
    Sobel test: z = a*b / sqrt(b^2 * se_a^2 + a^2 * se_b^2).

    :param X: Treatment/exposure variable (1-D array).
    :param M: Mediator variable (1-D array).
    :param Y: Outcome variable (1-D array).
    :return: Dictionary with total_effect, direct_effect, indirect_effect,
        proportion_mediated, sobel_z, sobel_p, a_path, b_path.
    :raises ValueError: If arrays have different lengths or fewer than 4 obs.

    References
    ----------
    Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
    distinction in social psychological research. *Journal of Personality
    and Social Psychology*, 51(6), 1173--1182.

    Sobel, M. E. (1982). Asymptotic confidence intervals for indirect
    effects in structural equation models. *Sociological Methodology*,
    13, 290--312.
    """
    from scipy import stats as _st

    Xv = np.asarray(X, dtype=float).ravel()
    Mv = np.asarray(M, dtype=float).ravel()
    Yv = np.asarray(Y, dtype=float).ravel()
    n = len(Xv)
    if len(Mv) != n or len(Yv) != n:
        raise ValueError("X, M, Y must have the same length.")
    if n < 4:
        raise ValueError("Need at least 4 observations.")

    def _ols(x_mat, y_vec):
        """Fit OLS, return coefficients and standard errors."""
        beta = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
        resid = y_vec - x_mat @ beta
        dof = len(y_vec) - x_mat.shape[1]
        mse = float(resid @ resid) / max(dof, 1)
        se = np.sqrt(np.diag(mse * np.linalg.inv(x_mat.T @ x_mat)))
        return beta, se

    ones = np.ones((n, 1))

    # Path 1: Y = c * X + intercept  (total effect)
    Xd1 = np.column_stack([ones, Xv])
    b1, _ = _ols(Xd1, Yv)
    c_total = float(b1[1])

    # Path 2: M = a * X + intercept
    Xd2 = np.column_stack([ones, Xv])
    b2, se2 = _ols(Xd2, Mv)
    a_path = float(b2[1])
    se_a = float(se2[1])

    # Path 3: Y = c' * X + b * M + intercept
    Xd3 = np.column_stack([ones, Xv, Mv])
    b3, se3 = _ols(Xd3, Yv)
    c_prime = float(b3[1])  # direct effect
    b_path = float(b3[2])
    se_b = float(se3[2])

    indirect = a_path * b_path

    # Proportion mediated
    prop_med = indirect / c_total if abs(c_total) > 1e-12 else float("nan")

    # Sobel test
    sobel_se = math.sqrt(b_path**2 * se_a**2 + a_path**2 * se_b**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else float("nan")
    sobel_p = float(2.0 * (1.0 - _st.norm.cdf(abs(sobel_z)))) if np.isfinite(sobel_z) else float("nan")

    return {
        "total_effect": c_total,
        "direct_effect": c_prime,
        "indirect_effect": indirect,
        "proportion_mediated": prop_med,
        "sobel_z": sobel_z,
        "sobel_p": sobel_p,
        "a_path": a_path,
        "b_path": b_path,
    }


medtn = causal_mediation


def cheatsheet() -> str:
    return "causal_mediation({}) -> Causal mediation analysis (Baron-Kenny approach)."
