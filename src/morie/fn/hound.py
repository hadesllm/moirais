# morie.fn — function file (hadesllm/morie)
"""Kalman filter tracking. 'I never lose a trail.' -- Hound"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kalman_filter(
    observations: np.ndarray,
    *,
    F: np.ndarray | None = None,
    H: np.ndarray | None = None,
    Q: float | np.ndarray = 1.0,
    R: float | np.ndarray = 1.0,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> DescriptiveResult:
    """1-D or multi-dimensional Kalman filter.

    Implements the standard predict-update Kalman filter.

    Parameters
    ----------
    observations : ndarray of shape (T,) or (T, m)
        Observed measurements.
    F : ndarray, optional
        State transition matrix. Defaults to identity.
    H : ndarray, optional
        Observation matrix. Defaults to identity.
    Q : float or ndarray
        Process noise covariance (scalar for 1-D).
    R : float or ndarray
        Measurement noise covariance (scalar for 1-D).
    x0 : ndarray, optional
        Initial state. Defaults to zeros.
    P0 : ndarray, optional
        Initial covariance. Defaults to identity.

    Returns
    -------
    DescriptiveResult
        With ``value`` = filtered state estimates and
        ``extra`` containing covariance history.
    """
    z = np.asarray(observations, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    T, m = z.shape

    if F is None:
        F = np.eye(m)
    else:
        F = np.asarray(F, dtype=float)
    n_state = F.shape[0]

    if H is None:
        H = np.eye(m, n_state)
    else:
        H = np.asarray(H, dtype=float)

    Q_mat = Q * np.eye(n_state) if np.isscalar(Q) else np.asarray(Q, dtype=float)
    R_mat = R * np.eye(m) if np.isscalar(R) else np.asarray(R, dtype=float)

    x = np.zeros(n_state) if x0 is None else np.asarray(x0, dtype=float).ravel()
    P = np.eye(n_state) if P0 is None else np.asarray(P0, dtype=float)

    filtered = np.zeros((T, n_state))
    residuals = np.zeros((T, m))

    for t in range(T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q_mat

        y_res = z[t] - H @ x_pred
        S = H @ P_pred @ H.T + R_mat
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y_res
        P = (np.eye(n_state) - K @ H) @ P_pred

        filtered[t] = x
        residuals[t] = y_res

    if m == 1:
        filtered = filtered.squeeze(axis=-1) if filtered.shape[1] == 1 else filtered

    return DescriptiveResult(
        name="kalman_filter",
        value=filtered,
        extra={"residuals": residuals, "n_obs": T, "n_state": n_state},
    )


hound = kalman_filter


def cheatsheet() -> str:
    return "kalman_filter({}) -> Kalman filter tracking. 'I never lose a trail.' -- Hound"
