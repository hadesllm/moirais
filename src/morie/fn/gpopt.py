# morie.fn — function file (hadesllm/morie)
"""Gaussian process Bayesian optimization."""

from __future__ import annotations

__all__ = ["gp_optimize", "gpopt"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize as sp_minimize
from scipy.spatial.distance import cdist


def _rbf_kernel(X1, X2, length_scale=1.0, variance=1.0, cdf=None):
    """Squared exponential (RBF) kernel."""
    dists = cdist(X1, X2, metric="sqeuclidean")
    return variance * np.exp(-0.5 * dists / length_scale ** 2)


def _gp_predict(X_train, y_train, X_test, length_scale, variance, noise):
    """GP posterior mean and variance."""
    K = _rbf_kernel(X_train, X_train, length_scale, variance)
    K += noise * np.eye(len(X_train))
    K_s = _rbf_kernel(X_train, X_test, length_scale, variance)
    K_ss = _rbf_kernel(X_test, X_test, length_scale, variance)

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-6 * np.eye(len(K))
        L = np.linalg.cholesky(K)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    cov = K_ss - v.T @ v
    var = np.maximum(np.diag(cov), 1e-10)

    return mu, var


def gp_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: Union[list, np.ndarray],
    *,
    n_calls: int = 30,
    n_initial: int = 5,
    length_scale: float = 1.0,
    noise: float = 1e-6,
    seed: int = 42,
) -> dict[str, Any]:
    r"""
    Bayesian optimization using a Gaussian process surrogate.

    Uses Expected Improvement (EI) as the acquisition function:

    .. math::

        \\text{EI}(x) = (f_{\\min} - \\mu(x)) \\Phi(z) + \\sigma(x) \\phi(z)

    where z = (f_min - mu(x)) / sigma(x).

    Parameters
    ----------
    objective : callable
        Black-box function mapping (d,) array to scalar.
    bounds : array-like
        Parameter bounds (d, 2), each row [lower, upper].
    n_calls : int
        Total number of objective evaluations.
    n_initial : int
        Random evaluations before GP model is used.
    length_scale : float
        RBF kernel length scale.
    noise : float
        Observation noise variance.
    seed : int
        Random seed.

    Returns
    -------
    dict
        best_x : ndarray
        best_y : float
        x_history : ndarray (n_calls, d)
        y_history : ndarray (n_calls,)

    References
    ----------
    Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient
    global optimization of expensive black-box functions. *Journal
    of Global Optimization*, 13(4), 455--492.
    Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
    for Machine Learning*, MIT Press.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    rng = np.random.default_rng(seed)

    x_history = np.empty((n_calls, d))
    y_history = np.empty(n_calls)

    for i in range(min(n_initial, n_calls)):
        x = bounds[:, 0] + rng.uniform(size=d) * (bounds[:, 1] - bounds[:, 0])
        x_history[i] = x
        y_history[i] = objective(x)

    for i in range(n_initial, n_calls):
        X_obs = x_history[:i]
        y_obs = y_history[:i]

        y_mean = np.mean(y_obs)
        y_std_val = np.std(y_obs) + 1e-10
        y_norm = (y_obs - y_mean) / y_std_val

        f_min = np.min(y_norm)

        def neg_ei(x_cand):
            x_cand = np.atleast_2d(x_cand)
            mu, var = _gp_predict(X_obs, y_norm, x_cand, length_scale, 1.0, noise)
            sigma = np.sqrt(var)
            z = (f_min - mu) / (sigma + 1e-30)
            ei = (f_min - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            return -float(ei[0])

        best_ei = np.inf
        best_x_next = None
        n_restarts = 10

        for _ in range(n_restarts):
            x0 = bounds[:, 0] + rng.uniform(size=d) * (bounds[:, 1] - bounds[:, 0])
            result = sp_minimize(
                neg_ei, x0, bounds=list(bounds), method="L-BFGS-B"
            )
            if result.fun < best_ei:
                best_ei = result.fun
                best_x_next = result.x

        x_history[i] = best_x_next
        y_history[i] = objective(best_x_next)

    best_idx = int(np.argmin(y_history))

    return {
        "best_x": x_history[best_idx],
        "best_y": float(y_history[best_idx]),
        "x_history": x_history,
        "y_history": y_history,
    }


gpopt = gp_optimize


def cheatsheet() -> str:
    return "gp_optimize(objective, bounds) -> Gaussian process Bayesian optimization."
