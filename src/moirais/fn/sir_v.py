"""SIR model with vaccination."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sir_vaccination(
    beta: float = 0.3,
    gamma: float = 0.1,
    v: float = 0.005,
    S0: float = 0.99,
    I0: float = 0.01,
    R0_init: float = 0.0,
    t_max: float = 160.0,
    n_steps: int = 1000,
) -> SIRResult:
    """
    SIR model with a constant vaccination rate removing susceptibles.

    .. math::

        \\frac{dS}{dt} = -\\beta S I - v S, \\quad
        \\frac{dI}{dt} = \\beta S I - \\gamma I, \\quad
        \\frac{dR}{dt} = \\gamma I + v S

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    v : float
        Vaccination rate (fraction of S vaccinated per unit time).
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
    Keeling, M. J., & Rohani, P. (2008). *Modeling Infectious Diseases
    in Humans and Animals*. Princeton University Press, Ch. 8.
    """
    if beta <= 0 or gamma <= 0:
        raise ValueError("beta and gamma must be positive.")
    if v < 0:
        raise ValueError("Vaccination rate v must be non-negative.")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s, i, r = y
        ds = -beta * s * i - v * s
        di = beta * s * i - gamma * i
        dr = gamma * i + v * s
        return [ds, di, dr]

    sol = odeint(deriv, [S0, I0, R0_init], t)
    R0_val = beta / gamma

    return SIRResult(
        model="SIR-V",
        t=t,
        S=sol[:, 0],
        I=sol[:, 1],
        R=sol[:, 2],
        R0=R0_val,
        extra={"beta": beta, "gamma": gamma, "v": v},
    )


sir_v = sir_vaccination


def cheatsheet() -> str:
    return "sir_vaccination({}) -> SIR model with vaccination."
