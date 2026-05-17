# morie.fn -- function file (hadesllm/morie)
"""PID controller simulation."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "No man ever steps in the same river twice. -- Heraclitus"


def pid_controller(
    setpoint: float = 1.0,
    measured: float = 0.0,
    kp: float = 1.0,
    ki: float = 0.1,
    kd: float = 0.01,
    dt: float = 0.01,
    n_steps: int = 100,
    plant_gain: float = 1.0,
    plant_tau: float = 0.5,
    **kwargs,
) -> DescriptiveResult:
    r"""
    Simulate a discrete PID controller driving a first-order plant.

    The PID control law is:

    .. math::

        u(t) = K_p e(t) + K_i \\int_0^t e(\\tau)\\,d\\tau
               + K_d \\frac{de}{dt}

    The plant is a first-order system :math:`\\tau \\dot{y} + y = K u`.

    :param setpoint: Desired target value. Default 1.0.
    :param measured: Initial measured value. Default 0.0.
    :param kp: Proportional gain. Default 1.0.
    :param ki: Integral gain. Default 0.1.
    :param kd: Derivative gain. Default 0.01.
    :param dt: Time step in seconds. Default 0.01.
    :param n_steps: Number of simulation steps. Default 100.
    :param plant_gain: Plant DC gain K. Default 1.0.
    :param plant_tau: Plant time constant tau. Default 0.5.
    :return: DescriptiveResult with steady-state error and time series.
    :raises ValueError: If dt <= 0 or n_steps < 1.

    References
    ----------
    Astrom, K. J. & Murray, R. M. (2021). *Feedback Systems* (2nd ed.).
    Princeton University Press.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")

    t = np.arange(n_steps) * dt
    y = np.zeros(n_steps)
    u = np.zeros(n_steps)
    e = np.zeros(n_steps)

    y[0] = measured
    integral = 0.0
    prev_error = setpoint - measured

    for i in range(1, n_steps):
        e[i] = setpoint - y[i - 1]
        integral += e[i] * dt
        derivative = (e[i] - prev_error) / dt
        u[i] = kp * e[i] + ki * integral + kd * derivative
        dy = (plant_gain * u[i] - y[i - 1]) / plant_tau
        y[i] = y[i - 1] + dy * dt
        prev_error = e[i]

    ss_error = float(np.abs(e[-1]))
    overshoot = float(np.max(y) - setpoint) if np.max(y) > setpoint else 0.0

    return DescriptiveResult(
        name="pid_controller",
        value=ss_error,
        extra={
            "steady_state_error": ss_error,
            "overshoot": overshoot,
            "time": t,
            "output": y,
            "control_signal": u,
            "error": e,
            "kp": kp,
            "ki": ki,
            "kd": kd,
        },
    )


pid = pid_controller


def cheatsheet() -> str:
    return "pid_controller({}) -> PID controller simulation."
