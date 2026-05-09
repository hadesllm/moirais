# moirais.fn — function file (hadesllm/moirais)
"""Partially linear GCV bandwidth selection."""

from __future__ import annotations

import numpy as np


def plgcv(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    h_grid: np.ndarray | None = None,
    kernel: str = "gaussian",
    n_grid: int = 20,
) -> dict:
    r"""
    GCV bandwidth selection for the partially linear model.

    Selects the kernel bandwidth for the Robinson estimator by
    generalised cross-validation:

    .. math::

        \text{GCV}(h) = \frac{n^{-1} \sum (Y_i - X_i'\hat\beta_h
        - \hat{g}_h(Z_i))^2}{(1 - \text{tr}(S_h)/n)^2}

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Linear covariates (n, p).
    Z : np.ndarray
        Nonparametric covariate (n,).
    h_grid : np.ndarray or None
        Candidate bandwidths.
    kernel : str
        Kernel function.
    n_grid : int
        Number of candidates when ``h_grid`` is None.

    Returns
    -------
    dict
        ``h_opt``, ``gcv_scores``, ``h_grid``, ``n_obs``.

    References
    ----------
    Horowitz (2009). Ch 3, eq. 3.24.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("y, X, Z must have same n.")
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    from moirais.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    h_rot = _silverman_bw(Z)

    if h_grid is None:
        h_grid = np.exp(np.linspace(np.log(h_rot * 0.1), np.log(h_rot * 5), n_grid))

    gcv_scores = np.empty(len(h_grid))

    for idx, h in enumerate(h_grid):
        diff = Z[:, None] - Z[None, :]
        W = k_fn(diff / h)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
        S = W / row_sums

        ey = S @ y
        y_tilde = y - ey

        X_tilde = X - S @ X
        XtX = X_tilde.T @ X_tilde
        try:
            beta = np.linalg.solve(XtX, X_tilde.T @ y_tilde)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtX, X_tilde.T @ y_tilde, rcond=None)[0]

        g_hat = ey - (S @ X) @ beta + X @ beta
        resid = y - X @ beta - (S @ (y - X @ beta))
        mse = np.mean(resid**2)

        tr_S = np.trace(S)
        denom = (1 - tr_S / n) ** 2
        gcv_scores[idx] = mse / max(denom, 1e-15)

    best = int(np.argmin(gcv_scores))
    return {
        "h_opt": float(h_grid[best]),
        "gcv_scores": gcv_scores.tolist(),
        "h_grid": h_grid.tolist(),
        "n_obs": n,
    }


plgcv_fn = plgcv


def cheatsheet() -> str:
    return "plgcv({y, X, Z}) -> GCV bandwidth for partially linear model."
