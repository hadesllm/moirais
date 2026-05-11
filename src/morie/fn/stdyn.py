"""Dynamic spatio-temporal state-space model (Schabenberger & Gotway Ch 9)."""

import numpy as np


def stdyn(
    data: np.ndarray,
    *,
    transition: np.ndarray | None = None,
    obs_noise: float = 1.0,
    state_noise: float = 0.1,
) -> dict:
    """
    Fit a dynamic linear spatio-temporal model (state-space / Kalman).

    .. math::

        \\theta_t = G \\theta_{t-1} + w_t, \\quad w_t \\sim N(0, W)
        z_t = F \\theta_t + v_t, \\quad v_t \\sim N(0, V)

    Uses a Kalman filter to estimate latent states and produce
    filtered estimates.

    :param data: Observations (T, n).
    :param transition: State transition matrix G (n, n); defaults to identity.
    :param obs_noise: Observation noise variance.
    :param state_noise: State innovation variance.
    :return: dict with ``filtered_states``, ``filtered_cov``, ``log_likelihood``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic
    Models*. Springer.

    Schabenberger & Gotway (2005), Ch. 9.
    """
    data = np.asarray(data, dtype=float)
    T, n = data.shape
    if transition is None:
        G = np.eye(n)
    else:
        G = np.asarray(transition, dtype=float)
        if G.shape != (n, n):
            raise ValueError(f"transition must be ({n}, {n}), got {G.shape}.")

    V = obs_noise * np.eye(n)
    W = state_noise * np.eye(n)
    F = np.eye(n)

    theta = np.zeros(n)
    P = np.eye(n)

    filtered_states = np.zeros((T, n))
    log_lik = 0.0

    for t in range(T):
        theta_pred = G @ theta
        P_pred = G @ P @ G.T + W

        S = F @ P_pred @ F.T + V
        K = P_pred @ F.T @ np.linalg.inv(S)
        innov = data[t] - F @ theta_pred
        theta = theta_pred + K @ innov
        P = (np.eye(n) - K @ F) @ P_pred

        filtered_states[t] = theta
        sign, logdet = np.linalg.slogdet(S)
        log_lik += -0.5 * (n * np.log(2 * np.pi) + logdet + float(innov @ np.linalg.solve(S, innov)))

    return {
        "filtered_states": filtered_states,
        "filtered_cov": P,
        "log_likelihood": float(log_lik),
        "obs_noise": obs_noise,
        "state_noise": state_noise,
        "T": T,
        "n": n,
    }


stdyn_fn = stdyn


def cheatsheet() -> str:
    return "stdyn({}) -> Dynamic spatio-temporal state-space model (Kalman)."
