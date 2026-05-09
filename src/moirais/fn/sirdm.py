"""SIR with age-structured demographics (two age groups)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ._containers import SIRResult


def sir_age_demographics(
    beta_matrix: list[list[float]] | None = None,
    gamma: float = 0.1,
    mu: list[float] | None = None,
    S0: list[float] | None = None,
    I0: list[float] | None = None,
    R0_init: list[float] | None = None,
    t_max: float = 200.0,
    n_steps: int = 1000,
) -> SIRResult:
    """SIR model with two age groups and group-specific contact rates.

    Unlike ``sir_d.py`` (single-population vital dynamics), this model
    has age-structured transmission via a contact matrix.

    Parameters
    ----------
    beta_matrix : list[list[float]], optional
        2x2 transmission matrix. Default [[0.3, 0.1], [0.1, 0.2]].
    gamma : float
        Recovery rate (same for both groups).
    mu : list[float], optional
        Death rates per group. Default [0.01, 0.02].
    S0 : list[float], optional
        Initial susceptible per group. Default [0.49, 0.49].
    I0 : list[float], optional
        Initial infected per group. Default [0.005, 0.005].
    R0_init : list[float], optional
        Initial recovered per group. Default [0.0, 0.0].
    t_max : float
        Simulation duration.
    n_steps : int
        Number of time points.

    Returns
    -------
    SIRResult
        S, I, R arrays have shape (n_steps, 2).

    References
    ----------
    Hethcote, H. W. (2000). The mathematics of infectious diseases.
    SIAM Review, 42(4), 599-653.
    """
    if beta_matrix is None:
        beta_matrix = [[0.3, 0.1], [0.1, 0.2]]
    if mu is None:
        mu = [0.01, 0.02]
    if S0 is None:
        S0 = [0.49, 0.49]
    if I0 is None:
        I0 = [0.005, 0.005]
    if R0_init is None:
        R0_init = [0.0, 0.0]

    B = np.array(beta_matrix)
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    t = np.linspace(0, t_max, n_steps)

    def deriv(y, _t):
        s = y[:2]
        i = y[2:4]
        r = y[4:6]
        foi = B @ i
        ds = np.array([mu[j] - foi[j] * s[j] - mu[j] * s[j] for j in range(2)])
        di = np.array([foi[j] * s[j] - (gamma + mu[j]) * i[j] for j in range(2)])
        dr = np.array([gamma * i[j] - mu[j] * r[j] for j in range(2)])
        return np.concatenate([ds, di, dr])

    y0 = np.concatenate([S0, I0, R0_init])
    sol = odeint(deriv, y0, t)

    S_tot = sol[:, 0] + sol[:, 1]
    I_tot = sol[:, 2] + sol[:, 3]
    R_tot = sol[:, 4] + sol[:, 5]

    eigvals = np.linalg.eigvals(B / gamma)
    r0_val = float(np.max(np.abs(eigvals)))

    return SIRResult(
        model="SIR-AGE",
        t=t,
        S=S_tot,
        I=I_tot,
        R=R_tot,
        R0=r0_val,
        extra={"S_age": sol[:, :2], "I_age": sol[:, 2:4], "R_age": sol[:, 4:6]},
    )


sirdm = sir_age_demographics


def cheatsheet() -> str:
    return "sir_age_demographics({}) -> SIR model with age-structured demographics."
