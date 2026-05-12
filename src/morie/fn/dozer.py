# morie.fn -- function file (hadesllm/morie)
"""Out of chaos, comes order. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pid_tune(
    setpoint: float,
    process: np.ndarray | list,
    *,
    dt: float = 1.0,
    kp: float = 1.0,
    ki: float = 0.1,
    kd: float = 0.05,
    n_steps: int | None = None,
) -> DescriptiveResult:
    """Simulate a PID controller on a discrete process signal.

    Computes the PID control output u(t) = Kp*e + Ki*integral(e) + Kd*de/dt
    for a given setpoint and measured process variable.

    Parameters
    ----------
    setpoint : float
        Desired target value.
    process : array-like
        Measured process variable over time.
    dt : float
        Time step between measurements.
    kp, ki, kd : float
        Proportional, integral, derivative gains.
    n_steps : int or None
        Number of steps to simulate (defaults to len(process)).

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean absolute error; ``extra`` has control output
        history, error history, and performance metrics.
    """
    pv = np.asarray(process, dtype=np.float64)
    if n_steps is None:
        n_steps = len(pv)
    if n_steps < 2:
        raise ValueError("Need at least 2 time steps")

    errors = np.zeros(n_steps)
    control = np.zeros(n_steps)
    integral = 0.0
    prev_err = 0.0

    for t in range(n_steps):
        measured = pv[t] if t < len(pv) else pv[-1]
        err = setpoint - measured
        errors[t] = err
        integral += err * dt
        derivative = (err - prev_err) / dt if t > 0 else 0.0
        control[t] = kp * err + ki * integral + kd * derivative
        prev_err = err

    mae = float(np.mean(np.abs(errors)))
    ise = float(np.sum(errors**2) * dt)
    settling_idx = n_steps
    threshold = 0.02 * abs(setpoint) if setpoint != 0 else 0.02
    for k in range(n_steps - 1, -1, -1):
        if abs(errors[k]) > threshold:
            settling_idx = k + 1
            break

    return DescriptiveResult(
        name="PID Controller",
        value=mae,
        extra={
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "ise": ise,
            "settling_step": settling_idx,
            "setpoint": setpoint,
            "n_steps": n_steps,
            "final_error": float(errors[-1]),
            "control_output": control.tolist(),
        },
    )


dozer = pid_tune


def cheatsheet() -> str:
    return "pid_tune({}) -> PID controller tuning. 'Everyone falls the first time.' -- C"
