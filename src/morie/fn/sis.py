"""SIS compartmental model (no immunity)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sis_model(
    beta: float = 0.3,
    gamma: float = 0.1,
    S0: float = 0.99,
    I0: float = 0.01,
    t_max: float = 160.0,
    n_steps: int = 1000,
) -> SIRResult:
    r"""
    Susceptible-Infected-Susceptible model without acquired immunity.

    .. math::

        \\frac{dS}{dt} = -\\beta S I + \\gamma I, \\quad
        \\frac{dI}{dt} = \\beta S I - \\gamma I

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    S0, I0 : float
        Initial fractions (should sum to 1).
    t_max : float
        Duration of simulation.
    n_steps : int
        Number of time points.

    Returns
    -------
    SIRResult

    References
    ----------
    Hethcote, H. W. (2000). The mathematics of infectious diseases.
    *SIAM Review*, 42(4), 599-653.
    """
    if beta <= 0 or gamma <= 0:
        raise ValueError("beta and gamma must be positive.")
    if S0 + I0 > 1.0 + 1e-9 or S0 < 0 or I0 < 0:
        raise ValueError("S0, I0 must be non-negative and sum to <= 1.")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s, i = y
        ds = -beta * s * i + gamma * i
        di = beta * s * i - gamma * i
        return [ds, di]

    sol = odeint(deriv, [S0, I0], t)
    R0 = beta / gamma

    return SIRResult(
        model="SIS",
        t=t,
        S=sol[:, 0],
        I=sol[:, 1],
        R=None,
        R0=R0,
        extra={"beta": beta, "gamma": gamma},
    )


sis = sis_model


def cheatsheet() -> str:
    return "sis_model({}) -> SIS compartmental model (no immunity)."
