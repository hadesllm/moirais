"""SVEIR compartmental model (vaccination + exposed)."""

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sveir_model(
    beta: float,
    sigma: float,
    gamma: float,
    p: float,
    eta: float = 1.0,
    N: int = 1000,
    V0: int = 0,
    E0: int = 0,
    I0: int = 1,
    R0_init: int = 0,
    t_max: int = 200,
    dt: float = 1.0,
) -> SIRResult:
    """Simulate an SVEIR model with vaccination and exposed class.

    .. math::

        \\frac{dS}{dt} = -\\beta S I / N - p S

        \\frac{dV}{dt} = p S - (1 - \\eta) \\beta V I / N

        \\frac{dE}{dt} = \\beta S I / N + (1 - \\eta) \\beta V I / N - \\sigma E

        \\frac{dI}{dt} = \\sigma E - \\gamma I

        \\frac{dR}{dt} = \\gamma I

    Parameters
    ----------
    beta : float
        Transmission rate.
    sigma : float
        Rate of progression from E to I.
    gamma : float
        Recovery rate.
    p : float
        Vaccination rate (S -> V).
    eta : float, default 1.0
        Vaccine efficacy (1 = perfect, 0 = no protection).
    N : int, default 1000
        Total population.
    V0, E0, I0, R0_init : int
        Initial compartment sizes.
    t_max : int, default 200
        Simulation duration.
    dt : float, default 1.0
        Time step.

    Returns
    -------
    SIRResult
        With extra['V'] for vaccinated compartment.

    References
    ----------
    Gumel, A. B., et al. (2004). Modelling strategies for controlling SARS
    outbreaks. Proceedings of the Royal Society B, 271, 2223-2232.
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError("Vaccine efficacy eta must be in [0, 1].")
    if not (p >= 0.0):
        raise ValueError("Vaccination rate p must be non-negative."The man who moves a mountain begins by carrying away small stones. — Confucius"SVEIR",
        t=t,
        S=S_a,
        E=E_a,
        I=I_a,
        R=R_a,
        R0=float(r0_val),
        extra={"V": V_a},
    )


sveir = sveir_model


def cheatsheet() -> str:
    return "sveir_model({}) -> SVEIR model with vaccination and exposed class."
