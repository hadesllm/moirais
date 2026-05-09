# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Binary NP via kernel likelihood."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def bnpkl(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Binary response model via nonparametric kernel likelihood.

    Maximises the kernel-smoothed log-likelihood:

    .. math::

        \ell(\beta) = \sum_{i=1}^n \left[
        Y_i \log \hat{p}_{-i}(X_i'\beta) +
        (1-Y_i) \log(1 - \hat{p}_{-i}(X_i'\beta))
        \right]

    where :math:`\hat{p}_{-i}` is a leave-one-out kernel estimator.

    Parameters
    ----------
    y : np.ndarray
        Binary response (n,), values in {0, 1}.
    X : np.ndarray
        Covariates (n, p).
    bandwidth : float or None
        Kernel bandwidth.
    kernel : str
        Kernel function.

    Returns
    -------
    dict
        ``beta`` (normalised), ``prob_hat`` (fitted probabilities),
        ``log_likelihood``, ``n_obs``.

    References
    ----------
    Horowitz (2009). Ch 4.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same n.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0/1).")
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    from moirais.fn.nwker import _get_kernel, _silverman_bw

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
        p_hat = (K @ y) / denom
        p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
        ll = np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
        return -ll

    b0 = np.zeros(p)
    b0[0] = 1.0
    res = minimize(neg_ll, b0, method="L-BFGS-B",
                   options={"maxiter": 100})
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)

    idx = X @ beta
    h = bandwidth if bandwidth is not None else _silverman_bw(idx)
    diff = idx[:, None] - idx[None, :]
    K = k_fn(diff / h)
    np.fill_diagonal(K, 0.0)
    denom = K.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    prob_hat = np.clip((K @ y) / denom, 1e-10, 1 - 1e-10)
    ll = float(np.sum(y * np.log(prob_hat) + (1 - y) * np.log(1 - prob_hat)))

    return {
        "beta": beta.tolist(),
        "prob_hat": prob_hat.tolist(),
        "log_likelihood": ll,
        "n_obs": n,
    }


bnpkl_fn = bnpkl


def cheatsheet() -> str:
    return "bnpkl({y, X}) -> Binary NP kernel likelihood."
