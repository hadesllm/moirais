# morie.fn — function file (hadesllm/morie)
"""Impulse response function (IRF) for VAR models."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ["irfun", "impulse_response"]


def impulse_response(
    coef: np.ndarray,
    sigma_u: np.ndarray,
    horizon: int = 20,
    shock_var: int = 0,
) -> DescriptiveResult:
    """Compute the orthogonalised impulse response function of a VAR model.

    Uses the Cholesky decomposition of the residual covariance matrix to
    orthogonalise the shocks (Cholesky identification).

    The MA(∞) representation is: ``Y[t] = sum_{h=0}^{inf} Phi[h] @ u[t-h]``
    where the orthogonalised IRF is ``Theta[h] = Phi[h] @ P`` and ``P`` is the
    lower Cholesky factor of ``sigma_u`` (i.e. ``P @ P' = sigma_u``).

    Parameters
    ----------
    coef : array-like, shape (m, 1 + m*p)
        VAR coefficient matrix from :func:`var_model` (row = equation,
        columns = [intercept, A1_row_i, ..., Ap_row_i]).
    sigma_u : array-like, shape (m, m)
        Residual covariance matrix.
    horizon : int
        Number of periods ahead to compute.  Default 20.
    shock_var : int
        Index (0-based) of the variable receiving the unit shock.  Default 0.

    Returns
    -------
    DescriptiveResult
        value: float — IRF at horizon h=1 for the shocked variable.
        extra keys:
          'irf'      : ndarray (horizon+1, m) — response of each variable
                       to a one-standard-deviation shock in *shock_var*.
          'Phi'      : ndarray (horizon+1, m, m) — VAR MA coefficient matrices.
          'chol'     : ndarray (m, m) — Cholesky factor P.
          'horizon'  : int.
          'shock_var': int.

    Raises
    ------
    ValueError
        If coef is not 2-D or sigma_u is not conformable.

    References
    ----------
    Lütkepohl H. (2005). New Introduction to Multiple Time Series Analysis.
    Springer. Ch. 2.3.

    Sims C.A. (1980). Macroeconomics and reality.
    Econometrica, 48(1), 1-48.
    """
    coef = np.asarray(coef, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    if coef.ndim != 2:
        raise ValueError(f"coef must be 2-D, got shape {coef.shape}.")
    m = coef.shape[0]
    p = (coef.shape[1] - 1) // m  # number of lags
    if sigma_u.shape != (m, m):
        raise ValueError(f"sigma_u must be ({m}, {m}), got {sigma_u.shape}.")
    if not (0 <= shock_var < m):
        raise ValueError(f"shock_var={shock_var} out of range for m={m}.")

    # Extract AR coefficient matrices A1, ..., Ap (each m x m).
    A = []
    for j in range(p):
        A.append(coef[:, 1 + j * m : 1 + (j + 1) * m].copy())

    # Compute MA coefficient matrices Phi[h] via the recursion:
    #   Phi[0] = I_m
    #   Phi[h] = sum_{j=1}^{min(h,p)} A[j-1] @ Phi[h-j]
    Phi = np.zeros((horizon + 1, m, m))
    Phi[0] = np.eye(m)
    for h in range(1, horizon + 1):
        for j in range(1, min(h, p) + 1):
            Phi[h] += A[j - 1] @ Phi[h - j]

    # Cholesky factor P such that P @ P' = sigma_u.
    # Regularise if not positive definite.
    try:
        P = np.linalg.cholesky(sigma_u)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma_u)
        eigvals = np.maximum(eigvals, 1e-12)
        P = eigvecs @ np.diag(np.sqrt(eigvals))

    # Orthogonalised IRF: Theta[h] = Phi[h] @ P.
    # Column shock_var of Theta[h] gives responses to shock in shock_var.
    irf = np.zeros((horizon + 1, m))
    for h in range(horizon + 1):
        irf[h] = (Phi[h] @ P)[:, shock_var]

    return DescriptiveResult(
        name="impulse_response",
        value=float(irf[1, shock_var]) if horizon >= 1 else float(irf[0, shock_var]),
        extra={
            "irf": irf.copy(),
            "Phi": Phi.copy(),
            "chol": P.copy(),
            "horizon": horizon,
            "shock_var": shock_var,
        },
    )


irfun = impulse_response


def cheatsheet() -> str:
    return "impulse_response(coef, sigma_u, horizon=20, shock_var=0) -> Orthogonalised IRF."
