# morie.fn — function file (hadesllm/morie)
"""LIME: Local Interpretable Model-Agnostic Explanations.

LIME approximates any black-box model locally around an instance
with a sparse linear model, using exponential kernel weighting to
emphasise proximity.

References
----------
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust
you?" Explaining the predictions of any classifier.
*Proceedings of ACM SIGKDD*, 1135-1144.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["limex"]


def limex(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    X_train: np.ndarray,
    *,
    n_samples: int = 1000,
    kernel_width: float | None = None,
    n_features: int | None = None,
    seed: int = 0,
    alpha: float = 0.01,
) -> dict[str, Any]:
    r"""Explain a prediction via LIME (local linear surrogate).

    Samples perturbed instances around ``instance``, weights them by
    the exponential kernel:

    .. math::

        \pi(z, x) = \exp\!\left(-\frac{d(z, x)^2}{\sigma^2}\right)

    then fits a weighted ridge regression to get coefficients
    (importances) for each feature.

    Parameters
    ----------
    predict_fn : Callable
        Model function ``(n, p) -> (n,)``.
    instance : np.ndarray
        Observation to explain, shape ``(p,)``.
    X_train : np.ndarray
        Training data for normalisation, shape ``(n, p)``.
    n_samples : int
        Number of perturbed samples.
    kernel_width : float, optional
        Bandwidth :math:`\sigma`; defaults to
        :math:`\sqrt{p} \times 0.75`.
    n_features : int, optional
        Number of top features to retain in the explanation.
        ``None`` keeps all.
    seed : int
        Random seed.
    alpha : float
        Ridge regularisation parameter.

    Returns
    -------
    dict
        ``coefficients`` (feature importances, shape ``(p,)``),
        ``intercept``, ``local_r2`` (weighted R² of surrogate),
        ``weights`` (kernel weights, shape ``(n_samples,)``),
        ``p``, ``method``.

    References
    ----------
    Ribeiro et al. (2016). KDD 2016, 1135-1144.
    """
    instance = np.asarray(instance, dtype=float).ravel()
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-D.")
    p = len(instance)
    if X_train.shape[1] != p:
        raise ValueError("instance and X_train must have the same number of features.")

    if kernel_width is None:
        kernel_width = float(np.sqrt(p) * 0.75)

    rng = np.random.default_rng(seed)

    # Scale to [0,1] range using training data
    x_min = X_train.min(axis=0)
    x_range = X_train.max(axis=0) - x_min
    x_range[x_range < 1e-10] = 1.0

    instance_norm = (instance - x_min) / x_range

    # Generate perturbed samples in normalised space
    Z_norm = rng.uniform(0, 1, size=(n_samples, p))

    # Reconstruct original-scale perturbations
    Z_orig = Z_norm * x_range + x_min

    # Kernel weights based on distance in normalised space
    dists = np.sqrt(np.sum((Z_norm - instance_norm) ** 2, axis=1))
    kernel_weights = np.exp(-(dists**2) / (kernel_width**2))

    # Get model predictions on perturbed samples
    f_z = predict_fn(Z_orig)

    # Weighted ridge regression
    sqrtW = np.sqrt(kernel_weights)
    Z_w = Z_orig * sqrtW[:, None]
    f_w = f_z * sqrtW

    # Add intercept column
    Z_aug = np.column_stack([np.ones(n_samples), Z_orig])
    Z_aug_w = np.column_stack([sqrtW, Z_w])

    # Ridge: (Z^TWZ + alpha*I) beta = Z^TWf
    n_feat = Z_aug_w.shape[1]
    A = Z_aug_w.T @ Z_aug_w + alpha * np.eye(n_feat)
    A[0, 0] -= alpha               # do not regularise intercept
    b = Z_aug_w.T @ f_w
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]

    intercept = float(beta[0])
    coefficients = beta[1:]

    # Weighted R² of surrogate
    f_hat = Z_aug @ beta
    ss_res = float(np.sum(kernel_weights * (f_z - f_hat)**2))
    f_mean = float(np.average(f_z, weights=kernel_weights))
    ss_tot = float(np.sum(kernel_weights * (f_z - f_mean)**2))
    local_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Optionally retain only top n_features by absolute importance
    if n_features is not None and n_features < p:
        top_idx = np.argsort(np.abs(coefficients))[::-1][:n_features]
        coef_sparse = np.zeros(p)
        coef_sparse[top_idx] = coefficients[top_idx]
        coefficients = coef_sparse

    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "local_r2": float(local_r2),
        "weights": kernel_weights,
        "p": p,
        "method": "LIME",
    }


def cheatsheet() -> str:
    return "limex(predict_fn, instance, X_train) -> LIME local explanation (Ribeiro et al. 2016, KDD)."
