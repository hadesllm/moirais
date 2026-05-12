# morie.fn — function file (hadesllm/morie)
"""SEIR compartmental model (Susceptible-Exposed-Infected-Recovered)."""

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def seir_model(
    beta: float,
    sigma: float,
    gamma: float,
    N: int = 1000,
    E0: int = 0,
    I0: int = 1,
    R0_init: int = 0,
    t_max: int = 160,
    dt: float = 1.0,
) -> SIRResult:
    r"""Simulate an SEIR compartmental model.

    .. math::

        \\frac{dS}{dt} = -\\beta S I / N

        \\frac{dE}{dt} = \\beta S I / N - \\sigma E

        \\frac{dI}{dt} = \\sigma E - \\gamma I

        \\frac{dR}{dt} = \\gamma I

    Parameters
    ----------
    beta : float
        Transmission rate.
    sigma : float
        Rate of progression from exposed to infected (1/incubation period).
    gamma : float
        Recovery rate.
    N : int, default 1000
        Total population.
    E0 : int, default 0
        Initial exposed.
    I0 : int, default 1
        Initial infected.
    R0_init : int, default 0
        Initial recovered.
    t_max : int, default 160
        Simulation duration.
    dt : float, default 1.0
        Time step.

    Returns
    -------
    SIRResult
        With E array populated.

    References
    ----------
    Li, M. Y. & Muldowney, J. S. (1995). Global stability for the SEIR
    model in epidemiology. Mathematical Biosciences, 125(2), 155-164.
    """
    S0 = N - E0 - I0 - R0_init
    t = np.arange(0, t_max, dt)

    def deriv(y, _t, _N, _beta, _sigma, _gamma):
        S, E, I, R = y
        dSdt = -_beta * S * I / _N
        dEdt = _beta * S * I / _N - _sigma * E
        dIdt = _sigma * E - _gamma * I
        dRdt = _gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    y0 = [S0, E0, I0, R0_init]
    sol = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
    S_arr, E_arr, I_arr, R_arr = sol.T

    r0_val = beta / gamma if gamma > 0 else np.inf

    return SIRResult(
        model="SEIR",
        t=t,
        S=S_arr,
        E=E_arr,
        I=I_arr,
        R=R_arr,
        R0=float(r0_val),
    )


seir = seir_model


def cheatsheet() -> str:
    return "seir_model({}) -> SEIR compartmental model (Susceptible-Exposed-Infected-Recov"
