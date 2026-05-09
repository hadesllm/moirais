# moirais.fn — function file (hadesllm/moirais)
"""Kalman filter for state estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def kalman_fn(
    y: np.ndarray,
    A: float = 1.0,
    H: float = 1.0,
    Q: float = 0.01,
    R: float = 1.0,
) -> SignalResult:
    """Apply scalar Kalman filter to noisy observations.

    :param y: 1-D observation sequence.
    :param A: State transition coefficient (default 1.0).
    :param H: Observation model coefficient (default 1.0).
    :param Q: Process noise variance (default 0.01).
    :param R: Observation noise variance (default 1.0).
    :return: SignalResult with filtered state estimates.
    """
    from moirais._adaptive import kalman_filter

    y = np.asarray(y, dtype=float).ravel()
    x_est, P_est = kalman_filter(y, A=A, H=H, Q=Q, R=R)
    return SignalResult(
        name="kalman",
        filtered=x_est,
        n_samples=len(y),
        extra={"P_est": P_est},
    )


klmnf = kalman_fn


def cheatsheet() -> str:
    return "kalman_fn({}) -> Kalman filter for state estimation."
