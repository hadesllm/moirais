"""Targeting parameter (TMLE fluctuation step)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special

__all__ = ["tgtpr"]


def tgtpr(
    Y: np.ndarray,
    Q_init: np.ndarray,
    H: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    link: str = "logit",
) -> dict[str, Any]:
    r"""
    Compute the targeting (fluctuation) parameter epsilon for TMLE.

    Fits the univariate submodel:

    .. math::

        \text{logit}\,\hat{Q}^*(\varepsilon) =
        \text{logit}\,\hat{Q}^0 + \varepsilon \cdot H

    via score equation: :math:`\sum H_i (Y_i - \hat{Q}^*_i) = 0`.

    :param Y: Outcome vector, shape (n,).
    :param Q_init: Initial outcome model predictions, shape (n,).
    :param H: Clever covariate, shape (n,).
    :param max_iter: Maximum Newton iterations. Default 100.
    :param tol: Convergence tolerance. Default 1e-8.
    :param link: ``"logit"`` (default) or ``"identity"``.
    :return: Dict with ``epsilon``, ``Q_star``, ``converged``, ``n_iter``, ``n``.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 17. Springer.
    van der Laan & Rose (2011). *Targeted Learning*. Springer.
    """
    Y = np.asarray(Y, dtype=float)
    Q_init = np.asarray(Q_init, dtype=float)
    H = np.asarray(H, dtype=float)
    n = len(Y)

    epsilon = 0.0
    converged = False

    if link == "logit":
        Q_logit = special.logit(np.clip(Q_init, 1e-6, 1 - 1e-6))
        for it in range(max_iter):
            Q_star = special.expit(Q_logit + epsilon * H)
            Q_star = np.clip(Q_star, 1e-8, 1 - 1e-8)
            score = np.sum(H * (Y - Q_star))
            info = np.sum(H ** 2 * Q_star * (1 - Q_star))
            if abs(info) < 1e-12:
                break
            step = score / info
            epsilon += step
            if abs(step) < tol:
                converged = True
                break
        Q_final = special.expit(Q_logit + epsilon * H)
    elif link == "identity":
        for it in range(max_iter):
            Q_star = Q_init + epsilon * H
            score = np.sum(H * (Y - Q_star))
            info = np.sum(H ** 2)
            if abs(info) < 1e-12:
                break
            step = score / info
            epsilon += step
            if abs(step) < tol:
                converged = True
                break
        Q_final = Q_init + epsilon * H
    else:
        raise ValueError(f"link must be 'logit' or 'identity', got '{link}'.")

    return {
        "epsilon": float(epsilon),
        "Q_star": Q_final,
        "converged": converged,
        "n_iter": it + 1 if 'it' in dir() else 0,
        "n": n,
    }


def cheatsheet() -> str:
    return "tgtpr(Y, Q_init, H) -> TMLE fluctuation parameter."
