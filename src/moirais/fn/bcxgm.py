# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Box-Cox transformation model via GMM.

Estimates the Box-Cox transformation parameter :math:`\\lambda` and
regression coefficients :math:`\\beta` in

.. math::

    \\frac{Y^\\lambda - 1}{\\lambda} = X^\\top \\beta + \\varepsilon

via Generalized Method of Moments using instrument-based orthogonality
conditions.

References
----------
Box, G. E. P. & Cox, D. R. (1964). An analysis of transformations.
    *Journal of the Royal Statistical Society, Series B*, 26(2),
    211--252.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize


def _boxcox_transform(y: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-8:
        return np.log(y)
    return (y ** lam - 1.0) / lam


def bcxgm(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    lambda_init: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Box-Cox transformation model via GMM.

    Parameters
    ----------
    Y : np.ndarray
        Positive outcome vector, shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    lambda_init : float
        Starting value for :math:`\\lambda`.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict[str, Any]
        ``lambda_hat``, ``beta``, ``se_lambda``, ``se_beta``,
        ``sigma2``, ``n``, ``p``, ``method``.

    Raises
    ------
    ValueError
        If any :math:`Y \\le 0`.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if len(Y) != n:
        raise ValueError(f"Y length {len(Y)} != X rows {n}.")
    if np.any(Y <= 0):
        raise ValueError("Y must be strictly positive for Box-Cox.")

    Xd = np.column_stack([X, np.ones(n)])
    p_aug = Xd.shape[1]

    def _gmm_obj(params: np.ndarray) -> float:
        lam = params[0]
        beta = params[1:]
        y_t = _boxcox_transform(Y, lam)
        resid = y_t - Xd @ beta
        moments = Xd.T @ resid / n
        return float(moments @ moments)

    params0 = np.zeros(1 + p_aug)
    params0[0] = lambda_init
    beta_init = np.linalg.lstsq(
        Xd, _boxcox_transform(Y, lambda_init), rcond=None
    )[0]
    params0[1:] = beta_init

    res = optimize.minimize(
        _gmm_obj, params0, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8},
    )

    lambda_hat = float(res.x[0])
    beta = res.x[1:]

    y_t = _boxcox_transform(Y, lambda_hat)
    resid = y_t - Xd @ beta
    sigma2 = float(np.sum(resid ** 2) / max(n - p_aug - 1, 1))

    eps_lam = 1e-5
    g_plus = _gmm_obj(res.x + eps_lam * np.eye(len(res.x))[0])
    g_minus = _gmm_obj(res.x - eps_lam * np.eye(len(res.x))[0])
    hess_lam = (g_plus - 2 * res.fun + g_minus) / eps_lam ** 2
    se_lambda = float(np.sqrt(1.0 / max(abs(hess_lam) * n, 1e-12)))

    XtX_inv = np.linalg.inv(Xd.T @ Xd + 1e-10 * np.eye(p_aug))
    se_beta = np.sqrt(sigma2 * np.diag(XtX_inv))

    return {
        "lambda_hat": lambda_hat,
        "beta": beta[:-1],
        "intercept": float(beta[-1]),
        "se_lambda": se_lambda,
        "se_beta": se_beta[:-1],
        "sigma2": sigma2,
        "n": n,
        "p": p,
        "method": "BoxCox_GMM",
    }


bcxgm_fn = bcxgm


def cheatsheet() -> str:
    return "bcxgm(Y, X) -> Box-Cox transformation model via GMM (Box & Cox 1964)."
