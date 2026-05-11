"""SIR compartmental model (Susceptible-Infected-Recovered)."""

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sir_model(
    beta: float,
    gamma: float,
    N: int = 1000,
    I0: int = 1,
    R0_init: int = 0,
    t_max: int = 160,
    dt: float = 1.0,
) -> SIRResult:
    """Simulate an SIR compartmental model using scipy.integrate.odeint.

    .. math::

        \\frac{dS}{dt} = -\\beta S I / N

        \\frac{dI}{dt} = \\beta S I / N - \\gamma I

        \\frac{dR}{dt} = \\gamma I

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    N : int, default 1000
        Total population.
    I0 : int, default 1
        Initial infected.
    R0_init : int, default 0
        Initial recovered.
    t_max : int, default 160
        Simulation duration (time units).
    dt : float, default 1.0
        Time step.

    Returns
    -------
    SIRResult

    References
    ----------
    Kermack, W. O. & McKendrick, A. G. (1927). A contribution to the
    mathematical theory of epidemics. Proc. R. Soc. Lond. A, 115(772),
    700-721.
    """
    S0 = N - I0 - R0_init
    t = np.arange(0, t_max, dt)

    def deriv(y, _t, _N, _beta, _gamma):
        S, I, R = y
        dSdt = -_beta * S * I / _N
        dIdt = _beta * S * I / _N - _gamma * I
        dRdt = _gamma * I
        return [dSdt, dIdt, dRdt]

    y0 = [S0, I0, R0_init]
    sol = odeint(deriv, y0, t, args=(N, beta, gamma))
    S_arr, I_arr, R_arr = sol.T

    r0_val = beta / gamma if gamma > 0 else np.inf

    return SIRResult(
        model="SIR",
        t=t,
        S=S_arr,
        I=I_arr,
        R=R_arr,
        R0=float(r0_val),
    )


sird = sir_model


def cheatsheet() -> str:
    return "sir_model({}) -> SIR compartmental model (Susceptible-Infected-Recovered)."
