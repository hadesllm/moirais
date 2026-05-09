# moirais.fn — function file (hadesllm/moirais)
"""Cybernetic control system simulation. 'Booyah!' -- Cyborg"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pid_simulate(
    setpoint: float = 1.0,
    *,
    kp: float = 1.0,
    ki: float = 0.1,
    kd: float = 0.05,
    dt: float = 0.01,
    n_steps: int = 500,
    plant_tau: float = 1.0,
    disturbance: float = 0.0,
) -> DescriptiveResult:
    """Simulate a PID controller on a first-order plant.

    Models a first-order system dy/dt = (-y + u) / tau with a PID controller
    tracking a setpoint.  Returns the time series of output, error, and
    control signal.

    Parameters
    ----------
    setpoint : float
        Target value.
    kp, ki, kd : float
        Proportional, integral, derivative gains.
    dt : float
        Time step.
    n_steps : int
        Number of simulation steps.
    plant_tau : float
        Plant time constant.
    disturbance : float
        Constant disturbance added to plant input.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'time'``, ``'output'``, ``'error'``, ``'control'``.
    """
    if plant_tau <= 0:
        raise ValueError("plant_tau must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    t_arr = np.arange(n_steps) * dt
    y_arr = np.zeros(n_steps)
    e_arr = np.zeros(n_steps)
    u_arr = np.zeros(n_steps)
    integral = 0.0
    prev_err = setpoint
    y = 0.0
    for i in range(n_steps):
        err = setpoint - y
        integral += err * dt
        deriv = (err - prev_err) / dt if i > 0 else 0.0
        u = kp * err + ki * integral + kd * deriv + disturbance
        y += (-y + u) / plant_tau * dt
        y_arr[i] = y
        e_arr[i] = err
        u_arr[i] = u
        prev_err = err
    overshoot = float(max(0, (np.max(y_arr) - setpoint) / setpoint * 100)) if setpoint != 0 else 0.0
    settle_idx = n_steps
    for i in range(n_steps - 1, -1, -1):
        if abs(y_arr[i] - setpoint) > 0.02 * abs(setpoint or 1):
            settle_idx = i + 1
            break
    settling_time = float(settle_idx * dt) if settle_idx < n_steps else float("inf")
    ss_error = float(abs(setpoint - y_arr[-1]))
    return DescriptiveResult(
        name="PID control simulation",
        value={
            "time": t_arr.tolist(),
            "output": y_arr.tolist(),
            "error": e_arr.tolist(),
            "control": u_arr.tolist(),
        },
        extra={
            "overshoot_pct": round(overshoot, 2),
            "settling_time": settling_time,
            "steady_state_error": ss_error,
            "final_output": float(y_arr[-1]),
            "kp": kp,
            "ki": ki,
            "kd": kd,
        },
    )


cybrg = pid_simulate


def cheatsheet() -> str:
    return "pid_simulate({}) -> Cybernetic control system simulation. 'Booyah!' -- Cyborg"
