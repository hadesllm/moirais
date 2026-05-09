"""Transformation regression model."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["trreg"]


def trreg(
    y: np.ndarray,
    X: np.ndarray,
    *,
    transform: str = "box-cox",
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Fit a transformation regression model.

    Estimates :math:`h(Y) = X^T \beta + \varepsilon` where :math:`h` is a
    monotone transformation estimated from the data.

    For Box-Cox: :math:`h_\lambda(y) = (y^\lambda - 1)/\lambda` when
    :math:`\lambda \neq 0`, and :math:`\log(y)` when :math:`\lambda = 0`.

    :param y: Response vector, shape (n,). Must be positive for Box-Cox.
    :param X: Covariate matrix, shape (n, p).
    :param transform: ``"box-cox"`` (default) or ``"log"``.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``beta``, ``se``, ``ci_lower``, ``ci_upper``,
        ``lambda_`` (Box-Cox parameter), ``log_likelihood``, ``n``.
    :raises ValueError: If arrays are empty or y non-positive for Box-Cox.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 12. Springer.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    X_aug = np.column_stack([np.ones(n), X])

    if transform == "log":
        if np.any(y <= 0):
            raise ValueError("y must be positive for log transform.")
        z = np.log(y)
        beta = np.linalg.lstsq(X_aug, z, rcond=None)[0]
        resid = z - X_aug @ beta
        sigma2 = float(np.var(resid, ddof=p + 1))
        ll = float(-n / 2 * np.log(2 * np.pi * sigma2) - np.sum(resid ** 2) / (2 * sigma2) + np.sum(np.log(1.0 / y)))
        lambda_ = 0.0
    elif transform == "box-cox":
        if np.any(y <= 0):
            raise ValueError("y must be positive for Box-Cox transform.")

        def box_cox(y_val, lam):
            if abs(lam) < 1e-10:
                return np.log(y_val)
            return (y_val ** lam - 1.0) / lam

        def neg_ll(lam):
            z = box_cox(y, lam)
            beta_t = np.linalg.lstsq(X_aug, z, rcond=None)[0]
            resid = z - X_aug @ beta_t
            sigma2 = np.var(resid, ddof=0)
            if sigma2 < 1e-300:
                return 1e10
            return n / 2 * np.log(sigma2) - (lam - 1) * np.sum(np.log(y))

        res = optimize.minimize_scalar(neg_ll, bounds=(-2, 2), method="bounded")
        lambda_ = float(res.x)
        z = box_cox(y, lambda_)
        beta = np.linalg.lstsq(X_aug, z, rcond=None)[0]
        resid = z - X_aug @ beta
        sigma2 = float(np.var(resid, ddof=p + 1))
        ll = float(-n / 2 * np.log(2 * np.pi * sigma2) - np.sum(resid ** 2) / (2 * sigma2) + (lambda_ - 1) * np.sum(np.log(y)))
    else:
        raise ValueError(f"transform must be 'box-cox' or 'log', got '{transform}'.")

    try:
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
        se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0))
    except np.linalg.LinAlgError:
        se = np.full(p + 1, np.nan)

    z_val = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "beta": beta,
        "se": se,
        "ci_lower": beta - z_val * se,
        "ci_upper": beta + z_val * se,
        "lambda_": lambda_,
        "log_likelihood": ll,
        "n": n,
    }


def cheatsheet() -> str:
    return "trreg({y, X}) -> Transformation regression model."
