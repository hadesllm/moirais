"""Tree-structured Parzen Estimator (TPE) for Bayesian optimization."""

from __future__ import annotations

__all__ = ["tpe_minimize", "treep"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np
from scipy import stats


def tpe_minimize(
    objective: Callable[[np.ndarray], float],
    bounds: Union[list, np.ndarray],
    *,
    n_calls: int = 50,
    n_initial: int = 10,
    gamma: float = 0.25,
    n_candidates: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Tree-structured Parzen Estimator (TPE) for black-box minimization.

    Models P(x | y < y*) and P(x | y >= y*) as separate KDEs, then
    proposes candidates that maximize the ratio l(x)/g(x), which is
    proportional to the expected improvement.

    Parameters
    ----------
    objective : callable
        Function mapping parameter vector (d,) to scalar loss.
    bounds : array-like
        Parameter bounds (d, 2), each row [lower, upper].
    n_calls : int
        Total number of objective evaluations.
    n_initial : int
        Random evaluations before TPE kicks in.
    gamma : float
        Quantile threshold for splitting observations (default 0.25).
    n_candidates : int
        Number of candidates drawn from l(x) per iteration.
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
    Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011).
    Algorithms for hyper-parameter optimization. *NIPS*, 24.
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
        n_obs = i
        threshold_idx = max(1, int(gamma * n_obs))
        sorted_idx = np.argsort(y_history[:n_obs])
        good_idx = sorted_idx[:threshold_idx]
        bad_idx = sorted_idx[threshold_idx:]

        x_good = x_history[good_idx]
        x_bad = x_history[bad_idx] if len(bad_idx) > 0 else x_history[sorted_idx[-1:]]

        bw_good = max(0.1, 1.0 / np.sqrt(len(x_good)))
        bw_bad = max(0.1, 1.0 / np.sqrt(len(x_bad)))

        best_ratio = -np.inf
        best_candidate = None

        for _ in range(n_candidates):
            idx = rng.integers(len(x_good))
            candidate = x_good[idx] + bw_good * rng.standard_normal(d)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

            log_l = 0.0
            for j in range(d):
                log_l += np.log(
                    np.mean(stats.norm.pdf(candidate[j], x_good[:, j], bw_good)) + 1e-30
                )

            log_g = 0.0
            for j in range(d):
                log_g += np.log(
                    np.mean(stats.norm.pdf(candidate[j], x_bad[:, j], bw_bad)) + 1e-30
                )

            ratio = log_l - log_g
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate

        x_history[i] = best_candidate
        y_history[i] = objective(best_candidate)

    best_idx = int(np.argmin(y_history))

    return {
        "best_x": x_history[best_idx],
        "best_y": float(y_history[best_idx]),
        "x_history": x_history,
        "y_history": y_history,
    }


treep = tpe_minimize


def cheatsheet() -> str:
    return "tpe_minimize(objective, bounds) -> Tree-structured Parzen Estimator."
