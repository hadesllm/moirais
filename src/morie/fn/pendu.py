# morie.fn -- function file (hadesllm/morie)
"""We suffer more often in imagination than in reality. -- Seneca"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pendulum(
    theta0: float,
    omega0: float = 0.0,
    length: float = 1.0,
    g: float = 9.81,
    t_max: float = 10.0,
    dt: float = 0.01,
) -> DescriptiveResult:
    r"""
    Simulate a simple pendulum using the nonlinear ODE.

    .. math::

        \\ddot{\\theta} = -\\frac{g}{L} \\sin(\\theta)

    Integrated via fourth-order Runge-Kutta.

    :param theta0: Initial angular displacement (radians).
    :param omega0: Initial angular velocity (rad/s). Default 0.
    :param length: Pendulum length (m). Default 1.0.
    :param g: Gravitational acceleration (m/s^2). Default 9.81.
    :param t_max: Simulation duration (s). Default 10.
    :param dt: Time step (s). Default 0.01.
    :return: DescriptiveResult with period estimate and time series.
    :raises ValueError: If length <= 0 or dt <= 0.

    References
    ----------
    Marion, J. B., & Thornton, S. T. (2004). *Classical Dynamics of
    Particles and Systems*. 5th ed. Brooks/Cole.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")

    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps + 1)
    theta = np.zeros(n_steps + 1)
    omega = np.zeros(n_steps + 1)
    theta[0] = theta0
    omega[0] = omega0

    ratio = g / length

    def f_omega(th: float) -> float:
        return -ratio * np.sin(th)

    for i in range(n_steps):
        k1_th = omega[i]
        k1_om = f_omega(theta[i])
        k2_th = omega[i] + 0.5 * dt * k1_om
        k2_om = f_omega(theta[i] + 0.5 * dt * k1_th)
        k3_th = omega[i] + 0.5 * dt * k2_om
        k3_om = f_omega(theta[i] + 0.5 * dt * k2_th)
        k4_th = omega[i] + dt * k3_om
        k4_om = f_omega(theta[i] + dt * k3_th)

        theta[i + 1] = theta[i] + (dt / 6) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = omega[i] + (dt / 6) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    small_angle_period = 2 * np.pi * np.sqrt(length / g)

    return DescriptiveResult(
        name="Simple Pendulum",
        value=float(small_angle_period),
        extra={
            "t": t,
            "theta": theta,
            "omega": omega,
            "small_angle_period": float(small_angle_period),
            "theta0": theta0,
            "length": length,
        },
    )


short = pendulum


def cheatsheet() -> str:
    return "We suffer more often in imagination than in reality. -- Seneca"
