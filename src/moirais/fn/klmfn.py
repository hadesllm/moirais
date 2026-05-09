# moirais.fn — function file (hadesllm/moirais)
"""Kalman filter wrapper for adaptive state estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def kalman_filter_fn(
    y: np.ndarray,
    A: np.ndarray | float = 1.0,
    H: np.ndarray | float = 1.0,
    Q: np.ndarray | float = 0.01,
    R: np.ndarray | float = 1.0,
    x0: float = 0.0,
    P0: float = 1.0,
) -> SignalResult:
    """Apply Kalman filter for optimal state estimation.

    :param y: 1-D observation sequence.
    :param A: State transition (scalar or matrix, default 1.0).
    :param H: Observation model (scalar or matrix, default 1.0).
    :param Q: Process noise variance (default 0.01).
    :param R: Measurement noise variance (default 1.0).
    :param x0: Initial state estimate (default 0.0).
    :param P0: Initial error covariance (default 1.0).
    :return: SignalResult with filtered state estimates.
    """
    from moirais._adaptive import kalman_filter

    y = np.asarray(y, dtype=float).ravel()
    x_est, P_est = kalman_filter(y, A=A, H=H, Q=Q, R=R, x0=x0, P0=P0)
    return SignalResult(
        name="kalman_filter",
        filtered=x_est,
        fs=1.0,
        n_samples=len(x_est),
        extra={"state_estimates": x_est, "error_covariance": P_est},
    )


klmfn = kalman_filter_fn


def cheatsheet() -> str:
    return "kalman_filter_fn({}) -> Kalman filter wrapper for adaptive state estimation."
