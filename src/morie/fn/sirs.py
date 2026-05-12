"""SIRS compartmental model (waning immunity)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sirs_model(
    beta: float = 0.3,
    gamma: float = 0.1,
    xi: float = 0.01,
    S0: float = 0.99,
    I0: float = 0.01,
    R0_init: float = 0.0,
    t_max: float = 300.0,
    n_steps: int = 1000,
) -> SIRResult:
    r"""
    Susceptible-Infected-Recovered-Susceptible model with waning immunity.

    .. math::

        \\frac{dS}{dt} = -\\beta S I + \\xi R, \\quad
        \\frac{dI}{dt} = \\beta S I - \\gamma I, \\quad
        \\frac{dR}{dt} = \\gamma I - \\xi R

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    xi : float
        Rate of immunity loss (waning).
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
    Hethcote, H. W. (2000). The mathematics of infectious diseases.
    *SIAM Review*, 42(4), 599-653.
    """
    if beta <= 0 or gamma <= 0 or xi <= 0:
        raise ValueError("beta, gamma, xi must be positive.")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s, i, r = y
        ds = -beta * s * i + xi * r
        di = beta * s * i - gamma * i
        dr = gamma * i - xi * r
        return [ds, di, dr]

    sol = odeint(deriv, [S0, I0, R0_init], t)
    R0_val = beta / gamma

    return SIRResult(
        model="SIRS",
        t=t,
        S=sol[:, 0],
        I=sol[:, 1],
        R=sol[:, 2],
        R0=R0_val,
        extra={"beta": beta, "gamma": gamma, "xi": xi},
    )


sirs = sirs_model


def cheatsheet() -> str:
    return "sirs_model({}) -> SIRS compartmental model (waning immunity)."
