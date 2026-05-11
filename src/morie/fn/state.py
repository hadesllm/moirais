"""State-space model with Kalman filter."""

import numpy as np

from ._containers import DescriptiveResult


def kalman_filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> DescriptiveResult:
    """
    Kalman filter for a linear Gaussian state-space model.

    State equation: x_t = F x_{t-1} + w_t, w_t ~ N(0, Q)
    Observation:    y_t = H x_t + v_t,      v_t ~ N(0, R)

    :param y: (n,) or (n, m) observations.
    :param F: (k, k) state transition matrix.
    :param H: (m, k) observation matrix.
    :param Q: (k, k) state noise covariance.
    :param R: (m, m) observation noise covariance.
    :param x0: (k,) initial state. Default zeros.
    :param P0: (k, k) initial state covariance. Default identity.
    :return: DescriptiveResult with filtered states, covariances, log-lik.
    :raises ValueError: On dimension mismatches.

    References
    ----------
    Kalman R.E. (1960). A new approach to linear filtering and
    prediction problems. *J. Basic Engineering*, 82(1), 35-45.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n, m = y.shape
    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    k = F.shape[0]
    if F.shape != (k, k):
        raise ValueError(f"F must be ({k},{k}), got {F.shape}.")
    if H.shape != (m, k):
        raise ValueError(f"H must be ({m},{k}), got {H.shape}.")
    if x0 is None:
        x0 = np.zeros(k)
    if P0 is None:
        P0 = np.eye(k)
    x = np.asarray(x0, dtype=float).ravel()
    P = np.asarray(P0, dtype=float)
    filtered_states = np.zeros((n, k))
    filtered_covs = np.zeros((n, k, k))
    log_lik = 0.0
    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innov = y[t] - H @ x_pred
        S = H @ P_pred @ H.T + R
        S_inv = np.linalg.inv(S)
        K = P_pred @ H.T @ S_inv
        x = x_pred + K @ innov
        P = (np.eye(k) - K @ H) @ P_pred
        filtered_states[t] = x
        filtered_covs[t] = P
        sign, logdet = np.linalg.slogdet(S)
        log_lik += -0.5 * (m * np.log(2 * np.pi) + logdet + float(innov @ S_inv @ innov))
    return DescriptiveResult(
        name="kalman_filter",
        value=float(log_lik),
        extra={
            "filtered_states": filtered_states,
            "filtered_covariances": filtered_covs,
            "log_likelihood": float(log_lik),
            "n": n,
            "state_dim": k,
            "obs_dim": m,
        },
    )


state = kalman_filter


def cheatsheet() -> str:
    return "kalman_filter({}) -> Kalman filter for state-space models."
