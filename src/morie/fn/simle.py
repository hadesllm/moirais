"""Single-index MLE."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def simle(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Single-index model via semiparametric maximum likelihood.

    Maximises the kernel-smoothed log-likelihood:

    .. math::

        \ell(\beta) = \sum_{i=1}^n \log \hat{G}_{-i}(X_i'\beta)^{Y_i}
        (1 - \hat{G}_{-i}(X_i'\beta))^{1-Y_i}

    where :math:`\hat{G}_{-i}` is a leave-one-out kernel estimator
    of :math:`P(Y=1|X'\beta)` (Klein & Spady 1993).

    Parameters
    ----------
    y : np.ndarray
        Binary response (n,), values in {0, 1}.
    X : np.ndarray
        Covariates (n, p), p >= 2.
    bandwidth : float or None
        Kernel bandwidth.
    kernel : str
        Kernel function.

    Returns
    -------
    dict
        ``beta`` (normalised), ``index``, ``g_hat`` (fitted
        probabilities), ``log_likelihood``, ``n_obs``.

    References
    ----------
    Klein, R. & Spady, R. (1993). An efficient semiparametric estimator
        for binary response models. Econometrica, 61, 387-421.
    Horowitz (2009). Ch 2 & 4.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}.")
    if p < 2:
        raise ValueError("Need p >= 2 covariates.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0/1).")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)

    def neg_ll(b):
        b_norm = b / (np.linalg.norm(b) + 1e-15)
        idx = X @ b_norm
        h = bandwidth if bandwidth is not None else _silverman_bw(idx)
        diff = idx[:, None] - idx[None, :]
        K = k_fn(diff / h)
        np.fill_diagonal(K, 0.0)
        denom = K.sum(axis=1)
        denom = np.where(denom < 1e-15, 1.0, denom)
        g = (K @ y) / denom
        g = np.clip(g, 1e-10, 1 - 1e-10)
        ll = np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))
        return -ll

    b0 = np.zeros(p)
    b0[0] = 1.0
    res = minimize(neg_ll, b0, method="L-BFGS-B",
                   options={"maxiter": 100, "ftol": 1e-8})
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)

    idx_final = X @ beta
    h = bandwidth if bandwidth is not None else _silverman_bw(idx_final)
    diff = idx_final[:, None] - idx_final[None, :]
    K = k_fn(diff / h)
    np.fill_diagonal(K, 0.0)
    denom = K.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    g_hat = (K @ y) / denom
    g_hat = np.clip(g_hat, 1e-10, 1 - 1e-10)
    ll = float(np.sum(y * np.log(g_hat) + (1 - y) * np.log(1 - g_hat)))

    return {
        "beta": beta.tolist(),
        "index": idx_final.tolist(),
        "g_hat": g_hat.tolist(),
        "log_likelihood": ll,
        "n_obs": n,
    }


simle_fn = simle


def cheatsheet() -> str:
    return "simle({y, X}) -> Single-index MLE (Klein-Spady)."
