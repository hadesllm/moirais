"""State-space model via Kalman filter."""

import numpy as np

from ._containers import DescriptiveResult


def state_space(y, F=None, H=None, Q=None, R_noise=None):
    """
    Fit a linear Gaussian state-space model via Kalman filter.

    Model: x_{t+1} = F x_t + w_t,  y_t = H x_t + v_t
    where w_t ~ N(0,Q), v_t ~ N(0,R).

    :param y: (T,) or (T,m) observations.
    :param F: (k,k) state transition. Defaults to [[1]].
    :param H: (m,k) observation matrix. Defaults to [[1]].
    :param Q: (k,k) process noise covariance.
    :param R_noise: (m,m) observation noise covariance.
    :return: DescriptiveResult with filtered states, log-likelihood.

    References
    ----------
    Durbin J, Koopman SJ (2012). Time Series Analysis by State Space
    Methods. 2nd ed. Oxford University Press.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    T, m = y.shape
    if F is None:
        F = np.eye(1)
    if H is None:
        H = np.eye(m, F.shape[0])
    k = F.shape[0]
    if Q is None:
        Q = np.eye(k) * 0.1
    if R_noise is None:
        R_noise = np.eye(m) * 1.0

    x = np.zeros(k)
    P = np.eye(k)
    states = np.zeros((T, k))
    ll = 0.0

    for t in range(T):
        v = y[t] - H @ x
        S = H @ P @ H.T + R_noise
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P @ H.T @ np.linalg.pinv(S)
        x = x + K @ v
        P = (np.eye(k) - K @ H) @ P
        states[t] = x
        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (m * np.log(2 * np.pi) + logdet + v @ np.linalg.solve(S, v))
        x = F @ x
        P = F @ P @ F.T + Q

    return DescriptiveResult(
        name="state_space",
        value=float(ll),
        extra={"filtered_states": states, "log_likelihood": float(ll), "T": T, "k": k},
    )


def cheatsheet() -> str:
    return "state_space({}) -> State-space model via Kalman filter."
