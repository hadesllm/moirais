# morie.fn — function file (hadesllm/morie)
"""Dynamic linear model (local level + local trend)."""

import numpy as np

from ._containers import DescriptiveResult


def dlm_fit(
    y: np.ndarray, model: str = "local_level", sigma2_obs: float = 1.0, sigma2_state: float = 0.1
) -> DescriptiveResult:
    """
    Dynamic linear model via Kalman filter.

    Supports 'local_level' (random walk + noise) and 'local_trend'
    (random walk with drift + noise).

    :param y: 1-D time series.
    :param model: 'local_level' or 'local_trend'. Default 'local_level'.
    :param sigma2_obs: Observation variance. Default 1.0.
    :param sigma2_state: State variance. Default 0.1.
    :return: DescriptiveResult with filtered level, trend (if applicable).
    :raises ValueError: If series too short or unknown model.

    References
    ----------
    West M. & Harrison J. (1997). Bayesian Forecasting and Dynamic
    Models, 2nd ed. Springer.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")
    if model == "local_level":
        k = 1
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[sigma2_state]])
    elif model == "local_trend":
        k = 2
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([sigma2_state, sigma2_state * 0.1])
    else:
        raise ValueError(f"model must be 'local_level' or 'local_trend', got '{model}'.")
    R = np.array([[sigma2_obs]])
    x = np.zeros(k)
    x[0] = y[0]
    P = np.eye(k) * 10
    states = np.zeros((n, k))
    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innov = y[t] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S[0, 0]
        x = x_pred + K.ravel() * innov[0]
        P = (np.eye(k) - K @ H) @ P_pred
        states[t] = x
    extra = {
        "level": states[:, 0],
        "model": model,
        "sigma2_obs": sigma2_obs,
        "sigma2_state": sigma2_state,
        "n": n,
    }
    if model == "local_trend":
        extra["trend"] = states[:, 1]
    return DescriptiveResult(
        name="dlm_fit",
        value=float(states[-1, 0]),
        extra=extra,
    )


dynic = dlm_fit


def cheatsheet() -> str:
    return "dlm_fit({}) -> Dynamic linear model (local level/trend)."
