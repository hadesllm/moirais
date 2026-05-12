# morie.fn -- function file (hadesllm/morie)
"""Kalman filter predict-update recursion (Kalman 1960)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["kalman_filter"]


def kalman_filter(x, F=None, H=None, Q=None, R=None,
                  x0=None, P0=None):
    r"""Run a generic linear Kalman filter.

    For each :math:`t`,

    .. math::

        \hat x_{t|t-1} &= F\,\hat x_{t-1|t-1},\\
        P_{t|t-1}    &= F\,P_{t-1|t-1}\,F' + Q,\\
        K_t           &= P_{t|t-1}\,H'(H\,P_{t|t-1}\,H' + R)^{-1},\\
        \hat x_{t|t} &= \hat x_{t|t-1} + K_t (y_t - H\,\hat x_{t|t-1}),\\
        P_{t|t}      &= (I - K_t H) P_{t|t-1}.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, m)
        Observations (m=1 if univariate). Univariate -> local-level
        defaults are used.
    F, H, Q, R, x0, P0 : optional
        State matrices; default to a univariate local-level model.

    Returns
    -------
    RichResult
        keys: ``state``, ``state_cov``, ``innovations``,
        ``innovation_variance``, ``loglik``, ``n``, ``method``.

    References
    ----------
    Kalman RE (1960). A New Approach to Linear Filtering and Prediction
    Problems. *J. Basic Eng.* 82(1), 35-45.
    """
    Y = np.asarray(x, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = Y.shape
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}.")

    if F is None:
        F = np.eye(m)
    if H is None:
        H = np.eye(m)
    if Q is None:
        Q = np.eye(m) * np.var(np.diff(Y, axis=0)) * 0.5
    if R is None:
        R = np.eye(m) * np.var(np.diff(Y, axis=0)) * 0.5
    F, H, Q, R = map(lambda a: np.atleast_2d(np.asarray(a, dtype=float)),
                     (F, H, Q, R))
    p = F.shape[0]
    if x0 is None:
        x0 = np.zeros(p)
        x0[: min(p, m)] = Y[0, : min(p, m)]
    if P0 is None:
        P0 = np.eye(p) * 1e6

    x_hat = np.zeros((n, p))
    P = np.zeros((n, p, p))
    innov = np.zeros((n, m))
    Sv = np.zeros((n, m, m))
    xc = np.asarray(x0, dtype=float).ravel()
    Pc = np.asarray(P0, dtype=float)
    ll = 0.0
    for t in range(n):
        xp = F @ xc
        Pp = F @ Pc @ F.T + Q
        v = Y[t] - H @ xp
        S = H @ Pp @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = Pp @ H.T @ Sinv
        xc = xp + K @ v
        Pc = (np.eye(p) - K @ H) @ Pp
        x_hat[t] = xc
        P[t] = Pc
        innov[t] = v
        Sv[t] = S
        sign, logdet = np.linalg.slogdet(S)
        if sign > 0:
            ll += -0.5 * (m * np.log(2 * np.pi) + logdet + v @ Sinv @ v)
    return RichResult(payload={
        "state": x_hat,
        "state_cov": P,
        "innovations": innov,
        "innovation_variance": Sv,
        "loglik": float(ll),
        "n": int(n),
        "method": "Linear Gaussian Kalman filter (numpy)",
    })


def cheatsheet():
    return "kalmn: Kalman filter (Kalman 1960)."
