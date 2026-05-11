# morie.fn — function file (hadesllm/morie)
"""SEIR with compartmental dynamics including vital dynamics and waning immunity."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def seir_compartmental(
    beta: float = 0.3,
    sigma: float = 0.2,
    gamma: float = 0.1,
    mu: float = 0.01,
    omega: float = 0.005,
    S0: float = 0.99,
    E0: float = 0.0,
    I0: float = 0.01,
    R0_init: float = 0.0,
    t_max: float = 300.0,
    n_steps: int = 1000,
) -> SIRResult:
    """SEIR model with vital dynamics (births/deaths) and waning immunity.

    .. math::

        \\frac{dS}{dt} = \\mu - \\beta S I - \\mu S + \\omega R

        \\frac{dE}{dt} = \\beta S I - (\\sigma + \\mu) E

        \\frac{dI}{dt} = \\sigma E - (\\gamma + \\mu) I

        \\frac{dR}{dt} = \\gamma I - (\\mu + \\omega) R

    Parameters
    ----------
    beta : float
        Transmission rate.
    sigma : float
        Rate E -> I (1 / incubation period).
    gamma : float
        Recovery rate.
    mu : float
        Birth = death rate (balanced vital dynamics).
    omega : float
        Waning immunity rate (R -> S).
    S0, E0, I0, R0_init : float
        Initial proportions.
    t_max : float
        Simulation duration.
    n_steps : int
        Number of time points.

    Returns
    -------
    SIRResult

    References
    ----------
    Li, M. Y. & Muldowney, J. S. (1995). Global stability for the SEIR
    model in epidemiology. Mathematical Biosciences, 125(2), 155-164.
    """
    if beta <= 0 or sigma <= 0 or gamma <= 0:
        raise ValueError("beta, sigma, gamma must be positive")
    if mu < 0 or omega < 0:
        raise ValueError("mu, omega must be non-negative")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s, e, i, r = y
        ds = mu - beta * s * i - mu * s + omega * r
        de = beta * s * i - (sigma + mu) * e
        di = sigma * e - (gamma + mu) * i
        dr = gamma * i - (mu + omega) * r
        return [ds, de, di, dr]

    sol = odeint(deriv, [S0, E0, I0, R0_init], t)
    r0_val = (beta * sigma) / ((sigma + mu) * (gamma + mu))

    return SIRResult(
        model="SEIR-C",
        t=t,
        S=sol[:, 0],
        E=sol[:, 1],
        I=sol[:, 2],
        R=sol[:, 3],
        R0=float(r0_val),
        extra={"beta": beta, "sigma": sigma, "gamma": gamma, "mu": mu, "omega": omega},
    )


seirc = seir_compartmental


def cheatsheet() -> str:
    return "seir_compartmental({}) -> SEIR with vital dynamics and waning immunity."
