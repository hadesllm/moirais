# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Backfitting algorithm for additive models.

Iteratively estimates each component :math:`m_j` of the additive model

.. math::

    Y = \\alpha + \\sum_{j=1}^p m_j(X_j) + \\varepsilon

by smoothing partial residuals :math:`Y - \\hat{\\alpha} -
\\sum_{k \\neq j} \\hat{m}_k(X_k)` against :math:`X_j`.

References
----------
Hastie, T. J. & Tibshirani, R. J. (1990). *Generalized Additive
    Models*. Chapman & Hall. Chapter 6.
Buja, A., Hastie, T., & Tibshirani, R. (1989). Linear smoothers and
    additive models. *Annals of Statistics*, 17(2), 453--510.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def bkfit(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, Any]:
    r"""Backfitting algorithm for additive models.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    bandwidth : float or None
        Kernel bandwidth for Nadaraya-Watson smoother; if *None*,
        uses Silverman's rule.
    max_iter : int
        Maximum backfitting iterations.
    tol : float
        Convergence tolerance (max change in any component).

    Returns
    -------
    dict[str, Any]
        ``intercept``, ``components`` (list of arrays, each ``(n,)``),
        ``fitted``, ``residuals``, ``iterations``, ``converged``,
        ``n``, ``p``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if len(Y) != n:
        raise ValueError(f"Y length {len(Y)} != X rows {n}.")

    if bandwidth is None:
        bandwidth = 1.06 * np.mean([np.std(X[:, j]) for j in range(p)]) * n ** (-1.0 / 5.0)

    def _smooth(x_col: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty(n)
        for i in range(n):
            w = np.exp(-0.5 * ((x_col - x_col[i]) / bandwidth) ** 2)
            ws = w.sum()
            out[i] = (w * y).sum() / ws if ws > 1e-12 else 0.0
        return out

    intercept = float(np.mean(Y))
    components = [np.zeros(n) for _ in range(p)]
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        max_change = 0.0
        for j in range(p):
            partial_resid = Y - intercept - sum(
                components[k] for k in range(p) if k != j
            )
            new_j = _smooth(X[:, j], partial_resid)
            new_j -= np.mean(new_j)
            change = np.max(np.abs(new_j - components[j]))
            if change > max_change:
                max_change = change
            components[j] = new_j

        intercept = float(np.mean(Y - sum(components)))

        if max_change < tol:
            converged = True
            break

    fitted = intercept + sum(components)
    residuals = Y - fitted

    return {
        "intercept": intercept,
        "components": components,
        "fitted": fitted,
        "residuals": residuals,
        "iterations": it,
        "converged": converged,
        "n": n,
        "p": p,
        "method": "Backfitting",
    }


bkfit_fn = bkfit


def cheatsheet() -> str:
    return "bkfit(Y, X) -> Backfitting algorithm for additive models (Hastie & Tibshirani 1990)."
