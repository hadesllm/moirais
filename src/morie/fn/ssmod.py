# morie.fn -- function file (hadesllm/morie)
"""Linear-Gaussian state-space model -- Kalman filter + smoother."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["state_space_model"]


def state_space_model(x):
    r"""Local-level state-space model fitted by Kalman filter.

    .. math::

        \mu_t = \mu_{t-1} + \eta_t,\quad
        y_t = \mu_t + \epsilon_t,\quad
        \eta_t\sim\mathcal{N}(0, Q),\ \epsilon_t\sim\mathcal{N}(0, R).

    Parameters
    ----------
    x : array-like
        Univariate observation series.

    Returns
    -------
    RichResult
        keys: ``filtered_state``, ``filtered_state_variance``,
        ``smoothed_state``, ``loglik``, ``Q`` (state-innovation variance),
        ``R`` (observation variance), ``n``, ``method``.

    References
    ----------
    Harvey AC (1989). *Forecasting, Structural Time Series Models and
    the Kalman Filter*. Cambridge UP.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")

    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        mod = UnobservedComponents(y, level="local level")
        fit = mod.fit(disp=False)
        Q = float(fit.params[0])
        R = float(fit.params[1])
        return RichResult(payload={
            "filtered_state": np.asarray(fit.filtered_state[0]),
            "filtered_state_variance": np.asarray(fit.filtered_state_cov[0, 0]),
            "smoothed_state": np.asarray(fit.smoothed_state[0]),
            "loglik": float(fit.llf),
            "Q": Q, "R": R, "n": int(n),
            "method": "Local-level Kalman via statsmodels.UnobservedComponents",
        })
    except Exception:
        pass

    # Pure-NumPy Kalman filter for local level (Q, R via simple MoM init).
    Q = float(np.var(np.diff(y))) / 2.0
    R = float(np.var(np.diff(y))) / 2.0
    a = np.zeros(n)
    P = np.zeros(n)
    a[0] = y[0]
    P[0] = 1e7
    ll = 0.0
    for t in range(1, n):
        a_pred = a[t - 1]
        P_pred = P[t - 1] + Q
        v = y[t] - a_pred
        F = P_pred + R
        K = P_pred / F
        a[t] = a_pred + K * v
        P[t] = P_pred - K * P_pred
        ll += -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F)
    # Backward smoother (RTS).
    a_s = a.copy()
    P_s = P.copy()
    for t in range(n - 2, -1, -1):
        Pp = P[t] + Q
        J = P[t] / Pp
        a_s[t] = a[t] + J * (a_s[t + 1] - a[t])
        P_s[t] = P[t] + J ** 2 * (P_s[t + 1] - Pp)
    return RichResult(payload={
        "filtered_state": a, "filtered_state_variance": P,
        "smoothed_state": a_s,
        "loglik": float(ll),
        "Q": float(Q), "R": float(R), "n": int(n),
        "method": "Local-level Kalman filter+smoother (numpy)",
    })


def cheatsheet():
    return "ssmod: Local-level Kalman state-space model (Harvey 1989)."
