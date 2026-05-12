# morie.fn -- function file (hadesllm/morie)
"""NOTEARS: DAG learning via continuous optimization.

NOTEARS reformulates DAG structure learning as a continuous
optimization problem using the algebraic characterization of
acyclicity: h(W) = tr(e^{W circ W}) - p = 0.

References
----------
Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018).
DAGs with NO TEARS: Continuous optimization for structure learning.
*Advances in Neural Information Processing Systems* (NeurIPS), 31.

Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. (2020).
Learning sparse nonparametric DAGs with NOTEARS. *AISTATS*, 3414-3425.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

__all__ = ["notrs"]


def notrs(
    X: np.ndarray,
    *,
    lambda1: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
) -> dict[str, Any]:
    r"""Learn a DAG via the NOTEARS continuous optimization algorithm.

    Solves the augmented Lagrangian problem:

    .. math::

        \min_{W} \frac{1}{2n}\|X - XW\|_F^2 + \lambda_1 \|W\|_1

        \text{s.t.} \; h(W) = \operatorname{tr}(e^{W \circ W}) - p = 0

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    lambda1 : float
        L1 regularisation strength (sparsity).
    max_iter : int
        Maximum augmented Lagrangian outer iterations.
    h_tol : float
        Convergence tolerance on :math:`h(W)`.
    rho_max : float
        Maximum penalty parameter.
    w_threshold : float
        Threshold for zeroing small weights in final DAG.

    Returns
    -------
    dict
        ``W`` (estimated weight matrix), ``dag`` (thresholded binary
        adjacency, ``dag[i,j]=1`` means i->j), ``h`` (acyclicity
        violation at convergence), ``p``, ``n``, ``method``.

    References
    ----------
    Zheng et al. (2018). NeurIPS 31.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n, p).")
    n, p = X.shape
    X_c = X - X.mean(axis=0)

    W = np.zeros((p, p))
    rho = 1.0
    alpha = 0.0           # Lagrange multiplier
    lr = 0.001

    def _loss(W_):
        resid = X_c - X_c @ W_
        return 0.5 / n * np.sum(resid**2)

    def _h(W_):
        with np.errstate(over="ignore"):
            WW = np.clip(W_ * W_, -50.0, 50.0)
        return float(np.trace(expm(WW)) - p)

    def _grad_loss(W_):
        resid = X_c - X_c @ W_
        return -X_c.T @ resid / n

    def _grad_h(W_):
        with np.errstate(over="ignore"):
            WW = np.clip(W_ * W_, -50.0, 50.0)
        E = np.clip(expm(WW), -1e15, 1e15)
        return 2.0 * W_ * E.T

    for _ in range(max_iter):
        # Gradient step on augmented Lagrangian
        for _inner in range(200):
            h_val = _h(W)
            gl = _grad_loss(W)
            gh = _grad_h(W)
            with np.errstate(over="ignore"):
                grad = gl + (alpha + rho * h_val) * gh
            W_new = W - lr * grad
            # Soft-threshold L1 on off-diagonal elements
            for i in range(p):
                for j in range(p):
                    if i != j:
                        W_new[i, j] = _soft_threshold(W_new[i, j], lr * lambda1)
                    else:
                        W_new[i, j] = 0.0
            W = W_new

        h_val = _h(W)
        alpha += rho * h_val
        rho = min(rho * 10.0, rho_max)
        if h_val <= h_tol:
            break

    # Threshold and enforce zero diagonal
    np.fill_diagonal(W, 0.0)
    dag = (np.abs(W) > w_threshold).astype(int)

    return {
        "W": W,
        "dag": dag,
        "h": float(_h(W)),
        "p": p,
        "n": n,
        "method": "NOTEARS",
    }


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def cheatsheet() -> str:
    return "notrs(X) -> NOTEARS DAG learning (Zheng et al. 2018, NeurIPS)."
