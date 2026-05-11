# morie.fn — function file (hadesllm/morie)
"""Counterfactual explanation via nearest counterfactual search.

Finds the smallest perturbation to an instance that changes the
model prediction, subject to feature constraints.

References
----------
Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual
explanations without opening the black box: Automated decisions and
the GDPR. *Harvard Journal of Law & Technology*, 31(2), 841-887.

Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine
learning classifiers through diverse counterfactual explanations.
*Proceedings of ACM FAccT*, 607-617.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["cfint"]


def cfint(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    X_train: np.ndarray,
    *,
    target_class: float = 1.0,
    n_cf: int = 5,
    lam_dist: float = 1.0,
    lam_feat: float = 0.5,
    max_iter: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    r"""Find counterfactual explanations via gradient-based optimisation.

    Minimises the Wachter loss:

    .. math::

        \mathcal{L}(x') =
            \lambda_1 \cdot (f(x') - y^*)^2
            + \lambda_2 \cdot \|x' - x\|_1

    from multiple random starting points to encourage diversity.

    Parameters
    ----------
    predict_fn : Callable
        Model function ``(n, p) -> (n,)`` returning a scalar per row
        (probability or score).
    instance : np.ndarray
        Instance to explain, shape ``(p,)``.
    X_train : np.ndarray
        Training set for initialisation and feasibility, shape ``(n, p)``.
    target_class : float
        Desired prediction target (e.g., 0.5 for a binary flip).
    n_cf : int
        Number of distinct counterfactuals to return.
    lam_dist : float
        Weight on distance penalty.
    lam_feat : float
        Weight on feature-wise MAD normalisation.
    max_iter : int
        Optimisation iterations per counterfactual.
    lr : float
        Gradient descent step size.
    seed : int
        Random seed for initialisation.

    Returns
    -------
    dict
        ``counterfactuals`` (shape ``(n_cf, p)``),
        ``distances`` (L1 distances, shape ``(n_cf,)``),
        ``predictions`` (model outputs, shape ``(n_cf,)``),
        ``instance``, ``target_class``, ``p``, ``method``.

    References
    ----------
    Wachter et al. (2017). Harvard JOLT, 31(2), 841-887.
    """
    instance = np.asarray(instance, dtype=float).ravel()
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-D.")
    p = len(instance)
    n_train = X_train.shape[0]

    rng = np.random.default_rng(seed)
    # Feature-wise MAD for normalisation
    mad = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0)
    mad[mad < 1e-8] = 1.0

    counterfactuals = []
    distances = []
    predictions = []

    # Multiple restarts from training set instances
    start_indices = rng.choice(n_train, size=min(n_cf * 3, n_train), replace=False)

    for idx in start_indices:
        if len(counterfactuals) >= n_cf:
            break
        x_cf = X_train[idx].copy()

        for _ in range(max_iter):
            # Numerical gradient of prediction loss
            pred = float(predict_fn(x_cf[None, :])[0])
            pred_loss = (pred - target_class) ** 2

            # Numerical gradient via finite differences
            eps = 1e-4
            grad = np.zeros(p)
            for j in range(p):
                x_plus = x_cf.copy()
                x_plus[j] += eps
                p_plus = float(predict_fn(x_plus[None, :])[0])
                grad[j] = 2.0 * (p_plus - target_class) * (p_plus - pred) / eps

            # L1 gradient (sign)
            dist_grad = lam_dist * np.sign(x_cf - instance) / mad

            x_cf -= lr * (grad + dist_grad)

        pred_final = float(predict_fn(x_cf[None, :])[0])
        dist_final = float(np.sum(np.abs(x_cf - instance) / mad))

        counterfactuals.append(x_cf.copy())
        distances.append(dist_final)
        predictions.append(pred_final)

    # Sort by distance
    order = np.argsort(distances)
    cfs = np.array(counterfactuals)[order][:n_cf]
    dists_out = np.array(distances)[order][:n_cf]
    preds_out = np.array(predictions)[order][:n_cf]

    return {
        "counterfactuals": cfs,
        "distances": dists_out,
        "predictions": preds_out,
        "instance": instance,
        "target_class": target_class,
        "p": p,
        "method": "counterfactual-Wachter",
    }


def cheatsheet() -> str:
    return "cfint(predict_fn, instance, X_train) -> Counterfactual explanation (Wachter et al. 2017)."
