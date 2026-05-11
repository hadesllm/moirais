"""Damped spring-mass system. 'Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spring_mass(
    m: float,
    k: float,
    x0: float,
    v0: float = 0.0,
    damping: float = 0.0,
    t_max: float = 10.0,
    dt: float = 0.01,
) -> DescriptiveResult:
    """
    Simulate a damped spring-mass system.

    .. math::

        m\\ddot{x} + c\\dot{x} + kx = 0

    Integrated via fourth-order Runge-Kutta.

    :param m: Mass (kg). Must be > 0.
    :param k: Spring constant (N/m). Must be > 0.
    :param x0: Initial displacement (m).
    :param v0: Initial velocity (m/s). Default 0.
    :param damping: Damping coefficient c (Ns/m). Default 0.
    :param t_max: Simulation duration (s). Default 10.
    :param dt: Time step (s). Default 0.01.
    :return: DescriptiveResult with natural frequency and time series.
    :raises ValueError: If m <= 0 or k <= 0.

    References
    ----------
    Rao, S. S. (2017). *Mechanical Vibrations*. 6th ed. Pearson.
    """
    if m <= 0:
        raise ValueError(f"m must be > 0, got {m}.")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")

    omega_n = np.sqrt(k / m)
    zeta = damping / (2 * np.sqrt(k * m))

    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0] = x0
    v[0] = v0

    def accel(pos: float, vel: float) -> float:
        return (-k * pos - damping * vel) / m

    for i in range(n_steps):
        k1x = v[i]
        k1v = accel(x[i], v[i])
        k2x = v[i] + 0.5 * dt * k1v
        k2v = accel(x[i] + 0.5 * dt * k1x, v[i] + 0.5 * dt * k1v)
        k3x = v[i] + 0.5 * dt * k2v
        k3v = accel(x[i] + 0.5 * dt * k2x, v[i] + 0.5 * dt * k2v)
        k4x = v[i] + dt * k3v
        k4v = accel(x[i] + dt * k3x, v[i] + dt * k3v)

        x[i + 1] = x[i] + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v[i + 1] = v[i] + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

    return DescriptiveResult(
        name="Damped Spring-Mass",
        value=float(omega_n),
        extra={
            "t": t,
            "x": x,
            "v": v,
            "omega_n": float(omega_n),
            "zeta": float(zeta),
            "damping_type": "underdamped" if zeta < 1 else ("critical" if zeta == 1 else "overdamped"),
        },
    )


short = spring_mass


def cheatsheet() -> str:
    return "spring_mass({}) -> Damped spring-mass system. 'Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu'"
