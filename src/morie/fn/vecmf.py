# morie.fn — function file (hadesllm/morie)
"""VECM estimation with error-correction (Johansen 1995)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["vecm"]


def vecm(Y, k_ar=1, coint_rank=1):
    r"""Estimate a Vector Error-Correction Model.

    .. math::

        \Delta Y_t = \alpha\beta' Y_{t-1}
                     + \sum_{i=1}^{p-1}\Gamma_i \Delta Y_{t-i} + \epsilon_t.

    Parameters
    ----------
    Y : array-like, shape (T, k)
        Panel of I(1) variables.
    k_ar : int, default 1
        Number of lagged differences (Γ-terms) in the VECM.
    coint_rank : int, default 1
        Cointegration rank ``r`` (number of stationary linear
        combinations).

    Returns
    -------
    RichResult
        keys: ``alpha`` (loading matrix, k × r), ``beta`` (cointegration
        vectors, k × r), ``Gamma`` (list of k × k lag matrices), ``Sigma``
        (residual covariance), ``loglik``, ``n``, ``k``, ``rank``,
        ``method``.

    References
    ----------
    Johansen S (1995). *Likelihood-Based Inference in Cointegrated Vector
    Autoregressive Models*. Oxford UP.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T
    T, k = Y.shape
    if T < 20 or k < 2 or coint_rank < 1 or coint_rank > k:
        raise ValueError(
            f"Need T>=20, k>=2, 1<=rank<=k; got T={T}, k={k}, r={coint_rank}.")

    try:
        from statsmodels.tsa.vector_ar.vecm import VECM
        m = VECM(Y, k_ar_diff=k_ar, coint_rank=coint_rank, deterministic="ci")
        fit = m.fit()
        return RichResult(payload={
            "alpha": np.asarray(fit.alpha),
            "beta":  np.asarray(fit.beta),
            "Gamma": [np.asarray(g) for g in fit.gamma.reshape(k_ar, k, k)]
                      if k_ar > 0 else [],
            "Sigma": np.asarray(fit.sigma_u),
            "loglik": float(fit.llf),
            "n": int(T), "k": int(k), "rank": int(coint_rank),
            "method": "VECM via statsmodels.tsa.vector_ar.vecm.VECM",
        })
    except Exception:
        pass

    # Pure-NumPy fallback (single-lag Γ_1 only; full Johansen reduced
    # rank via SVD on the Π matrix from OLS of ΔY on Y_{-1} and ΔY_{-1}).
    dY = np.diff(Y, axis=0)
    if k_ar == 0:
        Z0 = dY
        Z1 = Y[:-1]
        rows = Z0.shape[0]
        Pi_hat, *_ = np.linalg.lstsq(Z1, Z0, rcond=None)
        eps = Z0 - Z1 @ Pi_hat
    else:
        rows = dY.shape[0] - k_ar
        Z0 = dY[k_ar:]
        Z1 = Y[k_ar:-1] if k_ar > 0 else Y[:-1]
        Z2 = np.column_stack(
            [dY[k_ar - i - 1 : k_ar - i - 1 + rows] for i in range(k_ar)])
        X = np.column_stack([Z1, Z2])
        B, *_ = np.linalg.lstsq(X, Z0, rcond=None)
        Pi_hat = B[:k].T
        eps = Z0 - X @ B
    U, s, Vt = np.linalg.svd(Pi_hat.T, full_matrices=False)
    alpha = U[:, :coint_rank] * s[:coint_rank]
    beta = Vt[:coint_rank].T
    Sigma = (eps.T @ eps) / max(rows - 1, 1)
    return RichResult(payload={
        "alpha": alpha,
        "beta": beta,
        "Gamma": [],
        "Sigma": Sigma,
        "loglik": np.nan,
        "n": int(T), "k": int(k), "rank": int(coint_rank),
        "method": "VECM via SVD of OLS Π (numpy fallback)",
    })


def cheatsheet():
    return "vecmf: VECM estimation (Johansen 1995)."
