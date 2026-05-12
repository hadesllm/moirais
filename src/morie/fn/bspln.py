# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian spline regression."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_spline(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    n_knots: int = 10,
    degree: int = 3,
    alpha: float = 1.0,
    noise_var: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian spline regression with B-spline basis and conjugate prior.

    :param x: Input variable (n,).
    :param y: Response variable (n,).
    :param n_knots: Number of interior knots.
    :param degree: B-spline degree.
    :param alpha: Prior precision on spline coefficients.
    :param noise_var: Observation noise variance.
    :param prob: Credible interval probability.
    :return: Dictionary with fitted values, coefficients, credible bands.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(x_arr)

    knots = np.linspace(np.min(x_arr), np.max(x_arr), n_knots + 2)

    def _bspline_basis(t, knots, degree):
        p = len(knots) + degree - 1
        B = np.zeros((len(t), p))
        aug = np.concatenate([np.full(degree, knots[0]), knots, np.full(degree, knots[-1])])
        for j in range(p):
            for i, ti in enumerate(t):
                B[i, j] = _bspline_val(ti, j, degree, aug)
        return B

    def _bspline_val(t, j, k, knots):
        if k == 0:
            return 1.0 if knots[j] <= t < knots[j + 1] else 0.0
        d1 = knots[j + k] - knots[j]
        d2 = knots[j + k + 1] - knots[j + 1]
        c1 = (t - knots[j]) / d1 * _bspline_val(t, j, k - 1, knots) if d1 > 0 else 0.0
        c2 = (knots[j + k + 1] - t) / d2 * _bspline_val(t, j + 1, k - 1, knots) if d2 > 0 else 0.0
        return c1 + c2

    B = _bspline_basis(x_arr, knots, degree)
    p = B.shape[1]

    S_inv = alpha * np.eye(p) + B.T @ B / noise_var
    S = np.linalg.inv(S_inv)
    m = S @ B.T @ y_arr / noise_var

    fitted = B @ m
    pred_var = noise_var + np.sum(B @ S * B, axis=1)
    pred_sd = np.sqrt(pred_var)

    from scipy import stats as st
    z = st.norm.ppf(1 - (1 - prob) / 2)

    return {
        "fitted": fitted.tolist(),
        "coefficients": m.tolist(),
        "ci_lower": (fitted - z * pred_sd).tolist(),
        "ci_upper": (fitted + z * pred_sd).tolist(),
        "n_basis": p,
        "n_knots": n_knots,
    }


bspln = bayesian_spline


def cheatsheet() -> str:
    return "bayesian_spline({}) -> Bayesian spline regression."
