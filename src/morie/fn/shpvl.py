"""SHAP values via Kernel SHAP (model-agnostic).

Kernel SHAP uses a weighted linear model on the space of feature
coalitions to estimate Shapley values for any black-box model.

References
----------
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to
interpreting model predictions.
*Advances in Neural Information Processing Systems*, 30, 4768-4777.

Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn &
A. W. Tucker (Eds.), *Contributions to the Theory of Games*, Vol. 2.
Princeton University Press.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

__all__ = ["shpvl"]


def shpvl(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    background: np.ndarray,
    *,
    n_samples: int = 512,
    seed: int = 0,
) -> dict[str, Any]:
    r"""Estimate Shapley values via Kernel SHAP.

    For observation :math:`x`, finds :math:`\phi \in \mathbb{R}^p`
    such that:

    .. math::

        \hat{f}(x) \approx \phi_0 + \sum_{j=1}^p \phi_j

    by solving the Shapley-weighted regression:

    .. math::

        \min_\phi \sum_{z \in \{0,1\}^p}
            \pi(z) \left[ f(h(z, x)) - \phi_0 - \sum_j z_j \phi_j \right]^2

    where the Shapley kernel is
    :math:`\pi(z) = \frac{p-1}{\binom{p}{|z|}|z|(p-|z|)}`.

    Parameters
    ----------
    predict_fn : Callable
        Model function: ``(n, p) -> (n,)``.
    X : np.ndarray
        Observations to explain, shape ``(m, p)``.
    background : np.ndarray
        Background dataset for marginalisation, shape ``(b, p)``.
    n_samples : int
        Number of coalition samples per observation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``shap_values`` (shape ``(m, p)``),
        ``base_value`` (mean prediction over background),
        ``m``, ``p``, ``method``.

    References
    ----------
    Lundberg & Lee (2017). NeurIPS 30, 4768-4777.
    """
    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)
    if X.ndim != 2 or background.ndim != 2:
        raise ValueError("X and background must be 2-D.")
    m, p = X.shape
    if background.shape[1] != p:
        raise ValueError("X and background must have the same number of features.")

    rng = np.random.default_rng(seed)
    base_val = float(predict_fn(background).mean())

    shap_values = np.zeros((m, p))

    for i in range(m):
        xi = X[[i]]                         # shape (1, p)
        shap_values[i] = _kernel_shap_single(
            predict_fn, xi, background, base_val, p, n_samples, rng
        )

    return {
        "shap_values": shap_values,
        "base_value": base_val,
        "m": m,
        "p": p,
        "method": "KernelSHAP",
    }


def _kernel_shap_single(predict_fn, xi, bg, base_val, p, n_samples, rng):
    """Kernel SHAP for a single observation."""
    # Sample coalitions (exclude empty and full sets)
    n_z = min(n_samples, 2**p - 2)

    coalitions = []
    weights = []

    # Always include edge coalitions for stability
    for s in range(1, p):
        kern = (p - 1) / (comb(p, s) * s * (p - s))
        for combo in combinations(range(p), s):
            z = np.zeros(p, dtype=float)
            z[list(combo)] = 1.0
            coalitions.append(z)
            weights.append(kern)
        if len(coalitions) >= n_z:
            break

    # Supplement with random samples if needed
    while len(coalitions) < n_z:
        s = rng.integers(1, p)
        z = np.zeros(p)
        idx = rng.choice(p, size=s, replace=False)
        z[idx] = 1.0
        kern = (p - 1) / (comb(p, int(s)) * int(s) * (p - int(s)))
        coalitions.append(z)
        weights.append(kern)

    Z = np.array(coalitions)         # (n_z, p)
    w = np.array(weights)

    # Evaluate model on masked inputs (marginalise missing features)
    b = len(bg)
    f_z = np.empty(len(Z))
    for k, z_k in enumerate(Z):
        mask = z_k.astype(bool)
        # Replace masked-out features with background values
        X_masked = np.tile(xi, (b, 1))        # (b, p)
        X_masked[:, ~mask] = bg[:, ~mask]
        f_z[k] = predict_fn(X_masked).mean()

    # Shapley regression: weighted OLS (constrained: sum = f(x) - base)
    f_x = float(predict_fn(xi).mean())
    y = f_z - base_val

    W_diag = w
    ZtW = Z.T * W_diag
    A = ZtW @ Z + 1e-8 * np.eye(p)
    b_vec = ZtW @ y
    try:
        phi = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(A, b_vec, rcond=None)[0]

    # Efficiency correction: rescale so phi sums to f(x) - base
    s = phi.sum()
    if abs(s) > 1e-10:
        phi *= (f_x - base_val) / s

    return phi


def cheatsheet() -> str:
    return "shpvl(predict_fn, X, background) -> Kernel SHAP values (Lundberg & Lee 2017, NeurIPS)."
