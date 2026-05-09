# moirais.fn — function file (hadesllm/moirais)
"""Kalman filter (1-D state estimation).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 9.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

__all__ = ['kalmf']
def kalmf(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    A: float = 1.0,
    H: float = 1.0,
    Q: float = 1e-4,
    R: float = 1.0,
    x0: float = 0.0,
    P0: float = 1.0,
) -> SignalResult:
    """1-D Kalman filter for state estimation.

    State model: x[k] = A * x[k-1] + w, w ~ N(0, Q)
    Observation: z[k] = H * x[k] + v, v ~ N(0, R)

    Parameters
    ----------
    x : array-like
        1-D observation sequence.
    fs : float
        Sampling frequency in Hz.
    A : float
        State transition coefficient.
    H : float
        Observation coefficient.
    Q : float
        Process noise variance.
    R : float
        Measurement noise variance.
    x0 : float
        Initial state estimate.
    P0 : float
        Initial error covariance.

    Returns
    -------
    SignalResult
    """
    z = np.asarray(x, dtype=float).ravel()
    n = len(z)

    x_est = np.zeros(n)
    P_arr = np.zeros(n)
    K_arr = np.zeros(n)

    xk = x0
    Pk = P0

    for k in range(n):
        xp = A * xk
        Pp = A * Pk * A + Q
        K = Pp * H / (H * Pp * H + R)
        xk = xp + K * (z[k] - H * xp)
        Pk = (1 - K * H) * Pp
        x_est[k] = xk
        P_arr[k] = Pk
        K_arr[k] = K

    return SignalResult(
        name="kalmf",
        filtered=x_est,
        fs=fs,
        n_samples=n,
        extra={"covariance": P_arr, "kalman_gain": K_arr},
    )


def cheatsheet() -> str:
    return "kalmf({}) -> Kalman filter (1-D)."
