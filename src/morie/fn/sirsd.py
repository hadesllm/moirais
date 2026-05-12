"""SIRS model with demographics (births and deaths)."""

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sirs_demographics(
    beta: float,
    gamma: float,
    xi: float,
    mu: float = 0.0,
    nu: float | None = None,
    N: int = 1000,
    I0: int = 1,
    R0_init: int = 0,
    t_max: int = 300,
    dt: float = 1.0,
) -> SIRResult:
    r"""Simulate an SIRS model with vital dynamics (births and deaths).

    .. math::

        \\frac{dS}{dt} = \\nu N - \\beta S I / N + \\xi R - \\mu S

        \\frac{dI}{dt} = \\beta S I / N - \\gamma I - \\mu I

        \\frac{dR}{dt} = \\gamma I - \\xi R - \\mu R

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    xi : float
        Rate of immunity waning (R -> S).
    mu : float, default 0.0
        Natural death rate (per capita).
    nu : float or None
        Birth rate (per capita). Defaults to mu (constant population).
    N : int, default 1000
        Initial total population.
    I0, R0_init : int
        Initial infected and recovered.
    t_max : int, default 300
        Simulation duration.
    dt : float, default 1.0
        Time step.

    Returns
    -------
    SIRResult
        With R0 = beta / (gamma + mu).

    References
    ----------
    Hethcote, H. W. (2000). The mathematics of infectious diseases. SIAM
    Review, 42(4), 599-653.
    """
    if beta < 0 or gamma < 0 or xi < 0 or mu < 0:
        raise ValueError("All rate parameters must be non-negative.")
    if N <= 0:
        raise ValueError("Population N must be positive.")

    if nu is None:
        nu = mu
    S0 = N - I0 - R0_init
    t = np.arange(0, t_max, dt)

    def deriv(y, _t, _N, _b, _g, _x, _m, _n):
        S, I, R = y
        Nt = S + I + R
        dS = _n * Nt - _b * S * I / Nt + _x * R - _m * S
        dI = _b * S * I / Nt - _g * I - _m * I
        dR = _g * I - _x * R - _m * R
        return [dS, dI, dR]

    y0 = [S0, I0, R0_init]
    sol = odeint(deriv, y0, t, args=(N, beta, gamma, xi, mu, nu))
    S_a, I_a, R_a = sol.T

    r0_val = beta / (gamma + mu) if (gamma + mu) > 0 else np.inf

    return SIRResult(
        model="SIRS-D",
        t=t,
        S=S_a,
        I=I_a,
        R=R_a,
        R0=float(r0_val),
    )


sirsd = sirs_demographics


def cheatsheet() -> str:
    return "sirs_demographics({}) -> SIRS model with births and deaths."
