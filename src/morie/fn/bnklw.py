# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Binary Klein-Spady estimator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def bnklw(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    trimming: float = 0.01,
) -> dict:
    r"""
    Klein-Spady (1993) semiparametric binary response estimator.

    Maximises a trimmed kernel-smoothed log-likelihood:

    .. math::

        \ell(\beta) = \frac{1}{n} \sum_{i=1}^n \tau_i
        \left[Y_i \log \hat{p}_{-i} + (1-Y_i)\log(1-\hat{p}_{-i})\right]

    where :math:`\tau_i = \mathbf{1}[\hat{p}_{-i} \in [\delta, 1-\delta]]`
    trims extreme probabilities.

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
    trimming : float
        Trimming threshold delta in (0, 0.5).

    Returns
    -------
    dict
        ``beta`` (normalised), ``prob_hat``, ``log_likelihood``,
        ``n_trimmed``, ``n_obs``.

    References
    ----------
    Klein, R. & Spady, R. (1993). An efficient semiparametric estimator
        for binary response models. Econometrica, 61, 387-421.
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
    if not 0 < trimming < 0.5:
        raise ValueError("trimming must be in (0, 0.5).")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    delta = trimming

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
        trim = (p_hat >= delta) & (p_hat <= 1 - delta)
        p_c = np.clip(p_hat, 1e-10, 1 - 1e-10)
        ll = np.sum(trim * (y * np.log(p_c) + (1 - y) * np.log(1 - p_c)))
        return -ll

    b0 = np.zeros(p)
    b0[0] = 1.0
    res = minimize(neg_ll, b0, method="L-BFGS-B",
                   options={"maxiter": 200})
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)

    idx = X @ beta
    h = bandwidth if bandwidth is not None else _silverman_bw(idx)
    diff = idx[:, None] - idx[None, :]
    K = k_fn(diff / h)
    np.fill_diagonal(K, 0.0)
    denom = K.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    prob_hat = (K @ y) / denom
    trim = (prob_hat >= delta) & (prob_hat <= 1 - delta)
    prob_c = np.clip(prob_hat, 1e-10, 1 - 1e-10)
    ll = float(np.sum(trim * (y * np.log(prob_c) + (1 - y) * np.log(1 - prob_c))))

    return {
        "beta": beta.tolist(),
        "prob_hat": prob_hat.tolist(),
        "log_likelihood": ll,
        "n_trimmed": int((~trim).sum()),
        "n_obs": n,
    }


bnklw_fn = bnklw


def cheatsheet() -> str:
    return "bnklw({y, X}) -> Klein-Spady binary estimator."
