"""SIRS model with vaccination and waning immunity."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sirs_vaccination(
    beta: float = 0.3,
    gamma: float = 0.1,
    omega: float = 0.01,
    v: float = 0.005,
    S0: float = 0.99,
    I0: float = 0.01,
    R0_init: float = 0.0,
    t_max: float = 300.0,
    n_steps: int = 1000,
) -> SIRResult:
    """SIRS model with vaccination and waning immunity.

    Unlike SIR-V (``sir_v.py``), recovered individuals lose immunity at
    rate omega, returning to S. Vaccination moves S -> R directly.

    .. math::

        \\frac{dS}{dt} = -\\beta S I - v S + \\omega R

        \\frac{dI}{dt} = \\beta S I - \\gamma I

        \\frac{dR}{dt} = \\gamma I + v S - \\omega R

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    omega : float
        Waning immunity rate (R -> S).
    v : float
        Vaccination rate (S -> R).
    S0, I0, R0_init : float
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
    Keeling, M. J. & Rohani, P. (2008). *Modeling Infectious Diseases
    in Humans and Animals*. Princeton University Press, Ch. 2.
    """
    if beta <= 0 or gamma <= 0:
        raise ValueError("beta and gamma must be positive")
    if omega < 0 or v < 0:
        raise ValueError("omega and v must be non-negative")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s, i, r = y
        ds = -beta * s * i - v * s + omega * r
        di = beta * s * i - gamma * i
        dr = gamma * i + v * s - omega * r
        return [ds, di, dr]

    sol = odeint(deriv, [S0, I0, R0_init], t)
    r0_val = beta / gamma

    return SIRResult(
        model="SIRS-V",
        t=t,
        S=sol[:, 0],
        I=sol[:, 1],
        R=sol[:, 2],
        R0=float(r0_val),
        extra={"beta": beta, "gamma": gamma, "omega": omega, "v": v},
    )


sirsv = sirs_vaccination


def cheatsheet() -> str:
    return "sirs_vaccination({}) -> SIRS model with vaccination and waning immunity."
