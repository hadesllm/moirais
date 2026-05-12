# morie.fn -- function file (hadesllm/morie)
"""SEIQR compartmental model (Susceptible-Exposed-Infected-Quarantined-Recovered)."""

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def seiqr_model(
    beta: float,
    sigma: float,
    gamma: float,
    delta: float,
    kappa: float = 0.0,
    N: int = 1000,
    E0: int = 0,
    I0: int = 1,
    Q0: int = 0,
    R0_init: int = 0,
    t_max: int = 160,
    dt: float = 1.0,
) -> SIRResult:
    r"""Simulate an SEIQR compartmental model with quarantine.

    .. math::

        \\frac{dS}{dt} = -\\beta S I / N

        \\frac{dE}{dt} = \\beta S I / N - \\sigma E

        \\frac{dI}{dt} = \\sigma E - (\\gamma + \\delta) I

        \\frac{dQ}{dt} = \\delta I - \\kappa Q

        \\frac{dR}{dt} = \\gamma I + \\kappa Q

    Parameters
    ----------
    beta : float
        Transmission rate.
    sigma : float
        Rate of progression from E to I (1/incubation period).
    gamma : float
        Recovery rate from I (without quarantine).
    delta : float
        Quarantine rate (rate at which infected are quarantined).
    kappa : float, default 0.0
        Recovery rate from quarantine. If 0, defaults to gamma.
    N : int, default 1000
        Total population.
    E0, I0, Q0, R0_init : int
        Initial compartment sizes.
    t_max : int, default 160
        Simulation duration.
    dt : float, default 1.0
        Time step.

    Returns
    -------
    SIRResult
        With extra['Q'] array for quarantined compartment.

    References
    ----------
    Hethcote, H., Zhien, M., & Shengbing, L. (2002). Effects of quarantine
    in six endemic models for infectious diseases. Mathematical Biosciences,
    180(1-2), 141-160.
    """
    if beta < 0 or sigma < 0 or gamma < 0 or delta < 0:
        raise ValueError("All rate parameters must be non-negative.")
    if N <= 0:
        raise ValueError("Population N must be positive.")

    kappa = kappa if kappa > 0 else gamma
    S0 = N - E0 - I0 - Q0 - R0_init
    t = np.arange(0, t_max, dt)

    def deriv(y, _t, _N, _b, _s, _g, _d, _k):
        S, E, I, Q, R = y
        dS = -_b * S * I / _N
        dE = _b * S * I / _N - _s * E
        dI = _s * E - (_g + _d) * I
        dQ = _d * I - _k * Q
        dR = _g * I + _k * Q
        return [dS, dE, dI, dQ, dR]

    y0 = [S0, E0, I0, Q0, R0_init]
    sol = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, delta, kappa))
    S_a, E_a, I_a, Q_a, R_a = sol.T

    r0_val = beta / (gamma + delta) if (gamma + delta) > 0 else np.inf

    return SIRResult(
        model="SEIQR",
        t=t,
        S=S_a,
        E=E_a,
        I=I_a,
        R=R_a,
        R0=float(r0_val),
        extra={"Q": Q_a},
    )


seiqr = seiqr_model


def cheatsheet() -> str:
    return "seiqr_model({}) -> SEIQR compartmental model with quarantine."
