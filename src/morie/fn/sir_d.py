"""SIR model with vital dynamics (birth/death)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sir_demography(
    beta: float = 0.3,
    gamma: float = 0.1,
    mu: float = 0.01,
    S0: float = 0.99,
    I0: float = 0.01,
    R0_init: float = 0.0,
    t_max: float = 300.0,
    n_steps: int = 1000,
) -> SIRResult:
    """
    SIR model with birth and death (vital dynamics).

    .. math::

        \\frac{dS}{dt} = \\mu - \\beta S I - \\mu S, \\quad
        \\frac{dI}{dt} = \\beta S I - \\gamma I - \\mu I, \\quad
        \\frac{dR}{dt} = \\gamma I - \\mu R

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    mu : float
        Birth and death rate (assumes equal).
    S0, I0, R0_init : float
        Initial fractions.
    t_max : float
        Duration.
    n_steps : int
        Time points.

    Returns
    -------
    SIRResult

    References
    ----------
    Anderson, R. M., & May, R. M. (1991). *Infectious Diseases of
    Humans*. Oxford University Press, Ch. 3.
    """
    if beta <= 0 or gamma <= 0 or mu < 0:
        raise ValueError("beta, gamma must be positive; mu non-negative.")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s, i, r = y
        ds = mu - beta * s * i - mu * s
        di = beta * s * i - gamma * i - mu * i
        dr = gamma * i - mu * r
        return [ds, di, dr]

    sol = odeint(deriv, [S0, I0, R0_init], t)
    R0_val = beta / (gamma + mu)

    return SIRResult(
        model="SIR-D",
        t=t,
        S=sol[:, 0],
        I=sol[:, 1],
        R=sol[:, 2],
        R0=R0_val,
        extra={"beta": beta, "gamma": gamma, "mu": mu},
    )


sir_d = sir_demography


def cheatsheet() -> str:
    return "sir_demography({}) -> SIR model with vital dynamics (birth/death)."
