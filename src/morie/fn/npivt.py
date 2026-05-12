# morie.fn — function file (hadesllm/morie)
r"""Nonparametric IV via Tikhonov regularization.

Solves the Fredholm integral equation of the first kind:

.. math::

    E[Y \\mid Z] = \\int h(x) \\, f_{X|Z}(x \\mid z) \\, dx

by discretizing and applying Tikhonov regularization to the
ill-posed inverse problem :math:`A h = r`.

References
----------
Darolles, S., Fan, Y., Florens, J.-P., & Renault, E. (2011).
    Nonparametric instrumental regression. *Econometrica*, 79(5),
    1541--1565.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 7.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def npivt(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_basis: int = 20,
    alpha_reg: float | None = None,
    kernel_bw: float | None = None,
) -> dict[str, Any]:
    r"""Nonparametric IV via Tikhonov regularization.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    X : np.ndarray
        Endogenous regressor, shape ``(n,)``.
    Z : np.ndarray
        Instrument, shape ``(n,)``.
    n_basis : int
        Number of grid points for discretization.
    alpha_reg : float or None
        Tikhonov penalty; if *None*, uses :math:`n^{-1/3}`.
    kernel_bw : float or None
        Kernel bandwidth for density estimation.

    Returns
    -------
    dict[str, Any]
        ``x_grid``, ``h_hat`` (estimated structural function),
        ``alpha_reg``, ``n``, ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    n = len(Y)
    if len(X) != n or len(Z) != n:
        raise ValueError("Y, X, Z must have the same length.")

    if alpha_reg is None:
        alpha_reg = float(n ** (-1.0 / 3.0))
    if kernel_bw is None:
        kernel_bw = 1.06 * min(np.std(X), np.std(Z)) * n ** (-1.0 / 5.0)

    x_grid = np.linspace(X.min(), X.max(), n_basis)
    z_grid = np.linspace(Z.min(), Z.max(), n_basis)

    A = np.empty((n_basis, n_basis))
    r = np.empty(n_basis)
    for j, zj in enumerate(z_grid):
        wz = np.exp(-0.5 * ((Z - zj) / kernel_bw) ** 2)
        wz_sum = wz.sum()
        if wz_sum < 1e-12:
            wz = np.ones(n) / n
            wz_sum = 1.0
        r[j] = (wz * Y).sum() / wz_sum
        for k, xk in enumerate(x_grid):
            wx = np.exp(-0.5 * ((X - xk) / kernel_bw) ** 2)
            A[j, k] = (wz * wx).sum() / wz_sum

    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    A /= col_norms[None, :]

    AtA = A.T @ A
    h_hat = np.linalg.solve(
        AtA + alpha_reg * np.eye(n_basis), A.T @ r
    )
    h_hat /= col_norms

    return {
        "x_grid": x_grid,
        "h_hat": h_hat,
        "alpha_reg": alpha_reg,
        "n": n,
        "method": "NPIV_Tikhonov",
    }


npivt_fn = npivt


def cheatsheet() -> str:
    return "npivt(Y, X, Z) -> Nonparametric IV via Tikhonov regularization (Horowitz 2009)."
