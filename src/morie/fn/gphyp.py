# morie.fn -- function file (hadesllm/morie)
"""Compute negative marginal likelihood for GP hyperparameter optimization."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from ._containers import SpatialResult


def gaussian_process_marginal_likelihood(
    x_train: np.ndarray,
    y_train: np.ndarray,
    length_scale: float = 1.0,
    output_scale: float = 1.0,
    noise_var: float = 0.1,
) -> float:
    r"""
    Compute negative marginal likelihood for GP hyperparameter optimization.

    .. math::

        \log p(y | X, \theta) = -\frac{1}{2} \mathbf{y}^\top K^{-1} \mathbf{y}
        - \frac{1}{2} \log |K| - \frac{n}{2} \log 2\pi
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)

    n = len(y_train)

    sq_dist = np.sum((x_train[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    K = output_scale**2 * np.exp(-sq_dist / (2 * length_scale**2))
    K += noise_var * np.eye(n)

    try:
        L = cholesky(K, lower=True)
        alpha = solve_triangular(L, y_train, lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        log_det_K = 2 * np.sum(np.log(np.diag(L)))
        neg_mll = 0.5 * (y_train @ alpha + log_det_K + n * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        neg_mll = 1e10

    return neg_mll


def gphyp(
    data: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    n: int = 50,
    seed: int = 0,
    init_length_scale: float = 1.0,
    init_output_scale: float = 1.0,
    init_noise_var: float = 0.1,
    **kwargs,
) -> SpatialResult:
    """
    Optimize GP hyperparameters by maximizing marginal likelihood.

    :param data: (n,) training outputs. If None, synthetic data is generated.
    :param coords: (n, d) training inputs. If None, uniform-random coords are used.
    :param n: Number of synthetic points if data/coords not provided.
    :param seed: RNG seed for synthetic data.
    :param init_length_scale: Initial length scale.
    :param init_output_scale: Initial output scale.
    :param init_noise_var: Initial noise variance.
    :return: SpatialResult with optimized hyperparameters in .extra.
    """
    rng = np.random.default_rng(seed)
    if data is None:
        data = rng.standard_normal(n)
    data = np.asarray(data, dtype=float).ravel()
    n_points = len(data)

    if coords is None:
        coords = rng.uniform(0, 1, size=(n_points, 2))
    coords = np.asarray(coords, dtype=float)

    def objective(theta):
        return gaussian_process_marginal_likelihood(
            coords, data, theta[0], theta[1], theta[2]
        )

    init_theta = np.array([init_length_scale, init_output_scale, init_noise_var])
    opt = minimize(
        objective,
        init_theta,
        method='L-BFGS-B',
        bounds=[(0.01, 10), (0.01, 10), (0.001, 10)],
    )

    return SpatialResult(
        name="GP-Hyperparameters",
        statistic=float(np.mean(data)),
        extra={
            "length_scale": float(opt.x[0]),
            "output_scale": float(opt.x[1]),
            "noise_var": float(opt.x[2]),
            "mll": float(opt.fun),
            "success": bool(opt.success),
            "n_points": n_points,
        },
    )


def cheatsheet() -> str:
    return "gphyp(data=None, coords=None, n=50) -> SpatialResult with optimized GP hyperparameters"
