# morie.fn — function file (hadesllm/morie)
"""RKHS kernel ridge regression (Wahba 1990).

Solves the regularised problem

    min_f  (1/n) ||y - f(x)||^2 + lambda ||f||_H^2

with Gaussian kernel ``k(x,x') = exp(-||x-x'||^2 / (2 sigma^2))``.
Closed-form coefficients ``alpha = (K + n*lambda*I)^-1 y``; fitted
values ``yhat = K alpha``; in-sample SE from residual variance.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rkhs_kernel_regression"]


def rkhs_kernel_regression(x, y, sigma: float | None = None,
                           lam: float = 1e-3):
    """RKHS kernel ridge regression with Gaussian kernel.

    Parameters
    ----------
    x : (n,) or (n,d) array
    y : (n,) array
    sigma : float, optional
        Kernel bandwidth.  Default = median pairwise distance / sqrt(2).
    lam : float
        Ridge penalty (default 1e-3).

    Returns
    -------
    RichResult: alpha, fitted, residuals, sigma, lam, sse, se, r2, n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    if n < 2 or y.size != n:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "RKHS KRR (n<2)"})
    # pairwise distances
    diff = x[:, None, :] - x[None, :, :]
    D2 = np.sum(diff ** 2, axis=2)
    if sigma is None:
        med = np.median(np.sqrt(D2[D2 > 0])) if np.any(D2 > 0) else 1.0
        sigma = float(med / np.sqrt(2)) if med > 0 else 1.0
    K = np.exp(-D2 / (2 * sigma ** 2))
    alpha = np.linalg.solve(K + n * lam * np.eye(n), y)
    fitted = K @ alpha
    resid = y - fitted
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    se = float(np.sqrt(sse / max(1, n - 1)) / np.sqrt(n))
    return RichResult(payload={
        "alpha": alpha, "fitted": fitted, "residuals": resid,
        "sigma": float(sigma), "lambda": float(lam),
        "sse": sse, "r2": float(r2), "estimate": float(fitted.mean()),
        "se": se, "n": int(n),
        "method": "RKHS kernel ridge (Wahba 1990)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = np.linspace(0, 1, 50)
# >>> y = np.sin(2 * np.pi * x) + rng.normal(0, 0.05, 50)
# >>> res = rkhs_kernel_regression(x, y, lam=1e-4)
# >>> assert res["r2"] > 0.9


def cheatsheet():
    return "rkhsc(x, y, sigma=auto, lam=1e-3): RKHS kernel ridge regression."
