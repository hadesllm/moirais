"""Sieve likelihood estimation."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["slest"]


def slest(
    x: np.ndarray,
    *,
    n_basis: int | None = None,
    basis: str = "polynomial",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Sieve maximum likelihood density estimation.

    Approximates the log-density using a finite-dimensional sieve:

    .. math::

        \log f_K(x) = \sum_{k=0}^{K} \theta_k \phi_k(x) - A(\theta)

    where :math:`\phi_k` are basis functions and :math:`A(\theta)` is the
    normalizing constant.

    :param x: Observation array, shape (n,).
    :param n_basis: Number of basis functions. Default sqrt(n).
    :param basis: ``"polynomial"`` (default) or ``"cosine"``.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``coefficients``, ``n_basis``, ``log_likelihood``,
        ``eval_points``, ``density``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 14. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.size
    if n_basis is None:
        n_basis = max(2, int(np.sqrt(n)))

    x_std = (x - np.mean(x)) / max(np.std(x, ddof=0), 1e-10)

    if basis == "polynomial":
        def phi(z, K):
            return np.column_stack([z ** k for k in range(K)])
    elif basis == "cosine":
        def phi(z, K):
            return np.column_stack([np.cos(k * np.pi * z) for k in range(K)])
    else:
        raise ValueError(f"basis must be 'polynomial' or 'cosine', got '{basis}'.")

    Phi = phi(x_std, n_basis)

    theta = np.zeros(n_basis)
    lr = 0.01
    for _ in range(200):
        log_unnorm = Phi @ theta
        log_unnorm -= np.max(log_unnorm)
        weights = np.exp(log_unnorm)
        Z = np.mean(weights)
        grad = np.mean(Phi, axis=0) - np.mean(Phi * (weights / Z)[:, None], axis=0)
        theta += lr * grad
        if np.max(np.abs(grad)) < 1e-6:
            break

    eval_points = np.linspace(np.min(x_std), np.max(x_std), 200)
    Phi_eval = phi(eval_points, n_basis)
    log_density = Phi_eval @ theta
    log_density -= np.max(log_density)
    density = np.exp(log_density)
    density /= np.trapezoid(density, eval_points)

    log_unnorm = Phi @ theta
    log_unnorm -= np.max(log_unnorm)
    ll = float(np.mean(log_unnorm) - np.log(np.mean(np.exp(log_unnorm))))

    return {
        "coefficients": theta,
        "n_basis": n_basis,
        "log_likelihood": ll,
        "eval_points": eval_points * max(np.std(x, ddof=0), 1e-10) + np.mean(x),
        "density": density / max(np.std(x, ddof=0), 1e-10),
        "n": n,
    }


def cheatsheet() -> str:
    return "slest({x}) -> Sieve likelihood density estimation."
