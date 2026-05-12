# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
r"""Additive model via marginal integration.

Estimates an additive model :math:`m(x) = c + \\sum_j m_j(x_j)` using
marginal integration (Linton & Nielsen 1995): each component
:math:`m_j` is estimated by averaging the full (pilot) regression
surface over all other dimensions.

References
----------
Linton, O. & Nielsen, J. P. (1995). A kernel method of estimating
    structured nonparametric regression based on marginal integration.
    *Biometrika*, 82(1), 93--100.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def admod(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    grid_size: int = 50,
) -> dict[str, Any]:
    r"""Additive model via marginal integration.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    bandwidth : float or None
        Kernel bandwidth; if *None*, uses Silverman's rule.
    grid_size : int
        Number of grid points per component.

    Returns
    -------
    dict[str, Any]
        ``intercept``, ``components`` (list of dicts with ``x_grid``
        and ``m_hat``), ``residuals``, ``n``, ``p``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if len(Y) != n:
        raise ValueError(f"Y length {len(Y)} != X rows {n}.")

    if bandwidth is None:
        bandwidth = 1.06 * np.mean([np.std(X[:, j]) for j in range(p)]) * n ** (-1.0 / (4 + p))

    def _nw_1d(x_col: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        m = np.empty(len(x_eval))
        for i, xe in enumerate(x_eval):
            w = np.exp(-0.5 * ((x_col - xe) / bandwidth) ** 2)
            ws = w.sum()
            m[i] = (w * y).sum() / ws if ws > 1e-12 else 0.0
        return m

    intercept = float(np.mean(Y))
    components = []

    for j in range(p):
        x_grid = np.linspace(X[:, j].min(), X[:, j].max(), grid_size)

        m_j = np.empty(grid_size)
        for gi, xg in enumerate(x_grid):
            w = np.exp(-0.5 * ((X[:, j] - xg) / bandwidth) ** 2)
            ws = w.sum()
            if ws > 1e-12:
                m_j[gi] = (w * Y).sum() / ws
            else:
                m_j[gi] = intercept

        m_j -= np.mean(m_j)
        components.append({"x_grid": x_grid, "m_hat": m_j})

    fitted = intercept * np.ones(n)
    for j in range(p):
        comp = components[j]
        fitted += np.interp(X[:, j], comp["x_grid"], comp["m_hat"])
    residuals = Y - fitted

    return {
        "intercept": intercept,
        "components": components,
        "residuals": residuals,
        "n": n,
        "p": p,
        "method": "AdditiveModel_MarginalIntegration",
    }


admod_fn = admod


def cheatsheet() -> str:
    return "admod(Y, X) -> Additive model via marginal integration (Linton & Nielsen 1995)."
