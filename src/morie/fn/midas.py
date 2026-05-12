# morie.fn — function file (hadesllm/morie)
"""MIDAS mixed-frequency regression (Ghysels, Santa-Clara & Valkanov 2004)."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ._richresult import RichResult

__all__ = ["midas_regression"]


def _beta_weights(theta1, theta2, K):
    """Normalised Beta polynomial weights B(k; theta1, theta2), k=1..K."""
    k = np.arange(1, K + 1) / (K + 1)
    w = (k ** (theta1 - 1.0)) * ((1.0 - k) ** (theta2 - 1.0))
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def midas_regression(x, y, K=None):
    r"""MIDAS regression with Beta-polynomial weights.

    .. math::

        y_t = \beta_0 + \beta_1 \sum_{k=0}^{K-1} B(k+1;\theta_1,\theta_2)
              x_{t-k/m} + \epsilon_t

    Parameters
    ----------
    x : array-like, shape (n_t, K)
        High-frequency regressor: row t holds the K most recent
        high-frequency observations available at low-frequency time t
        (most recent first).  Pass a flat 1-D array if K is supplied to
        auto-reshape with stride 1.
    y : array-like, shape (n_t,)
        Low-frequency target.
    K : int, optional
        Number of high-frequency lags used if x is supplied flat.

    Returns
    -------
    RichResult
        keys: ``beta0``, ``beta1``, ``theta1``, ``theta2``, ``weights``,
        ``loglik``, ``r2``, ``n``, ``K``, ``method``.

    References
    ----------
    Ghysels E, Santa-Clara P, Valkanov R (2004). The MIDAS Touch:
    Mixed Data Sampling Regression Models. *CIRANO Working paper*.
    """
    Y = np.asarray(y, dtype=float).ravel()
    Xf = np.asarray(x, dtype=float)
    if Xf.ndim == 1:
        if K is None:
            raise ValueError("Pass K when x is a flat 1-D array.")
        nT = Y.size
        if Xf.size < K + nT - 1:
            raise ValueError(
                f"x too short: need >= K + n_y - 1 = {K + nT - 1}.")
        # Stack: row t = [x[nT-1-t+K-1] … x[nT-1-t]] (most recent first).
        rows = []
        for t in range(nT):
            end = Xf.size - (nT - 1 - t)
            rows.append(Xf[end - K:end][::-1])
        X = np.asarray(rows)
    else:
        X = Xf
        K = X.shape[1]
    nT, K = X.shape
    if Y.size != nT:
        raise ValueError(f"y has {Y.size} rows, x has {nT}.")
    if nT < 4:
        raise ValueError(f"Need at least 4 obs, got {nT}.")

    def neg_ll(p):
        b0, b1, t1, t2 = p
        if t1 <= 0 or t2 <= 0:
            return 1e10
        w = _beta_weights(t1, t2, K)
        yhat = b0 + b1 * (X @ w)
        resid = Y - yhat
        sse = float(np.sum(resid ** 2))
        return sse if np.isfinite(sse) else 1e10

    fit = optimize.minimize(
        neg_ll,
        [float(np.mean(Y)), 1.0, 1.5, 2.0],
        bounds=[(-1e3, 1e3), (-1e3, 1e3), (0.1, 50), (0.1, 50)],
        method="L-BFGS-B",
    )
    b0, b1, t1, t2 = fit.x
    w = _beta_weights(t1, t2, K)
    yhat = b0 + b1 * (X @ w)
    resid = Y - yhat
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return RichResult(payload={
        "beta0": float(b0), "beta1": float(b1),
        "theta1": float(t1), "theta2": float(t2),
        "weights": w,
        "loglik": -0.5 * nT * (np.log(2 * np.pi)
                               + np.log(np.var(resid) + 1e-12) + 1.0),
        "r2": float(r2),
        "n": int(nT), "K": int(K),
        "method": "MIDAS Beta-polynomial via L-BFGS-B (numpy)",
    })


def cheatsheet():
    return "midas: MIDAS mixed-frequency regression (Ghysels et al. 2004)."
