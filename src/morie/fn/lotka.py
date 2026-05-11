# morie.fn — function file (hadesllm/morie)
"""Lotka-Volterra predator-prey model."""

import numpy as np

from ._containers import DescriptiveResult
def lotka_volterra(
    prey0: float = 40.0,
    pred0: float = 9.0,
    alpha: float = 0.1,
    beta: float = 0.02,
    gamma: float = 0.3,
    delta: float = 0.01,
    t_max: float = 100.0,
    dt: float = 0.01,
    **kwargs,
) -> DescriptiveResult:
    """
    Simulate the Lotka-Volterra predator-prey ODE system.

    .. math::

        \\frac{dx}{dt} &= \\alpha x - \\beta x y \\\\
        \\frac{dy}{dt} &= \\delta x y - \\gamma y

    where *x* = prey, *y* = predator, alpha = prey growth rate,
    beta = predation rate, gamma = predator death rate,
    delta = predator reproduction rate.

    Solved with 4th-order Runge-Kutta.

    :param prey0: Initial prey population. Default 40.
    :param pred0: Initial predator population. Default 9.
    :param alpha: Prey intrinsic growth rate. Default 0.1.
    :param beta: Predation coefficient. Default 0.02.
    :param gamma: Predator mortality rate. Default 0.3.
    :param delta: Predator growth from predation. Default 0.01.
    :param t_max: Simulation duration. Default 100.
    :param dt: Time step. Default 0.01.
    :return: DescriptiveResult with equilibrium values and time series.

    References
    ----------
    Lotka, A. J. (1925). *Elements of Physical Biology*. Williams & Wilkins.
    Volterra, V. (1926). Variazioni e fluttuazioni del numero d'individui
    in specie animali conviventi. *Mem. Acad. Lincei*, 2, 31-113.
    """
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    prey = np.zeros(n_steps)
    pred = np.zeros(n_steps)
    prey[0] = prey0
    pred[0] = pred0

    def f(x, y):
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return dx, dy

    for i in range(n_steps - 1):
        x, y = prey[i], pred[i]
        k1x, k1y = f(x, y)
        k2x, k2y = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = f(x + dt * k3x, y + dt * k3y)
        prey[i + 1] = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        pred[i + 1] = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)

    eq_prey = gamma / delta if delta > 0 else float("inf")
    eq_pred = alpha / beta if beta > 0 else float("inf")

    return DescriptiveResult(
        name="lotka_volterra",
        value=float(np.mean(prey)),
        extra={
            "equilibrium_prey": eq_prey,
            "equilibrium_predator": eq_pred,
            "time": t,
            "prey": prey,
            "predator": pred,
            "max_prey": float(np.max(prey)),
            "max_predator": float(np.max(pred)),
            "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        },
    )


lotka = lotka_volterra


def cheatsheet() -> str:
    return "lotka_volterra({}) -> Lotka-Volterra predator-prey model."
