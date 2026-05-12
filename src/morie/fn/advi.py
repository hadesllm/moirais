# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Automatic differentiation variational inference (mean-field)."""

from __future__ import annotations

__all__ = ["advi_meanfield", "advi"]

from collections.abc import Callable
from typing import Any

import numpy as np


def advi_meanfield(
    log_target: Callable[[np.ndarray], float],
    dim: int,
    *,
    n_iter: int = 5000,
    n_samples: int = 10,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Mean-field ADVI using black-box variational inference.

    Approximates the posterior with a diagonal Gaussian
    q(theta) = N(mu, diag(exp(2 * omega))).  Gradients are
    estimated via the reparameterization trick with numerical
    differentiation of the log-target.

    Parameters
    ----------
    log_target : callable
        Log-density of target (unnormalized OK).
    dim : int
        Dimensionality of parameter space.
    n_iter : int
        Number of optimization steps.
    n_samples : int
        Monte Carlo samples per gradient estimate.
    learning_rate : float
        Step size for Adam optimizer.
    seed : int
        Random seed.

    Returns
    -------
    dict
        variational_mean : list of float
        variational_std : list of float
        elbo_history : list of float
        final_elbo : float
        n_iter : int

    References
    ----------
    Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M.
    (2017). Automatic differentiation variational inference. *JMLR*,
    18(14), 1--45.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1.")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")

    rng = np.random.default_rng(seed)

    mu = np.zeros(dim)
    omega = np.zeros(dim)

    m_mu = np.zeros(dim)
    v_mu = np.zeros(dim)
    m_om = np.zeros(dim)
    v_om = np.zeros(dim)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    elbo_history = []
    delta = 1e-5

    for t in range(1, n_iter + 1):
        sigma = np.exp(omega)

        g_mu = np.zeros(dim)
        g_om = np.zeros(dim)
        elbo_est = 0.0

        for _ in range(n_samples):
            eta = rng.standard_normal(dim)
            theta = mu + sigma * eta
            lp = log_target(theta)

            grad_lp = np.zeros(dim)
            for j in range(dim):
                theta_p = theta.copy()
                theta_p[j] += delta
                grad_lp[j] = (log_target(theta_p) - lp) / delta

            g_mu += grad_lp
            g_om += grad_lp * eta * sigma + 1.0

            entropy_contrib = np.sum(omega) + 0.5 * dim * np.log(2 * np.pi * np.e)
            elbo_est += lp + entropy_contrib

        g_mu /= n_samples
        g_om /= n_samples
        elbo_est /= n_samples
        elbo_history.append(float(elbo_est))

        m_mu = beta1 * m_mu + (1 - beta1) * g_mu
        v_mu = beta2 * v_mu + (1 - beta2) * g_mu ** 2
        m_hat_mu = m_mu / (1 - beta1 ** t)
        v_hat_mu = v_mu / (1 - beta2 ** t)
        mu += learning_rate * m_hat_mu / (np.sqrt(v_hat_mu) + eps_adam)

        m_om = beta1 * m_om + (1 - beta1) * g_om
        v_om = beta2 * v_om + (1 - beta2) * g_om ** 2
        m_hat_om = m_om / (1 - beta1 ** t)
        v_hat_om = v_om / (1 - beta2 ** t)
        omega += learning_rate * m_hat_om / (np.sqrt(v_hat_om) + eps_adam)

    return {
        "variational_mean": mu.tolist(),
        "variational_std": np.exp(omega).tolist(),
        "elbo_history": elbo_history,
        "final_elbo": elbo_history[-1] if elbo_history else float("nan"),
        "n_iter": n_iter,
    }


advi = advi_meanfield


def cheatsheet() -> str:
    return "advi_meanfield(log_target, dim) -> Mean-field ADVI."
