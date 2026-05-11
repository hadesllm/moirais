# morie.fn — function file (hadesllm/morie)
"""Klein-Gordon equation."""

__all__ = ["klgrd"]

import numpy as np
from scipy.integrate import solve_ivp


def klgrd(
    m: float,
    x_range: tuple = (-10.0, 10.0),
    n_x: int = 200,
    t_span: tuple = (0.0, 5.0),
    n_t: int = 100,
    initial_phi=None,
    initial_dphi_dt=None,
    hbar: float = 1.0,
    c: float = 1.0,
) -> dict:
    """
    Solve the 1+1D Klein-Gordon equation numerically.

    .. math::

        \\frac{1}{c^2} \\frac{\\partial^2 \\phi}{\\partial t^2}
        - \\frac{\\partial^2 \\phi}{\\partial x^2}
        + \\frac{m^2 c^2}{\\hbar^2} \\phi = 0

    Uses method of lines with centered finite differences in x.

    Parameters
    ----------
    m : float
        Mass parameter (natural units if hbar=c=1).
    x_range : tuple
        (x_min, x_max) spatial domain.
    n_x : int
        Number of spatial grid points.
    t_span : tuple
        (t_start, t_end).
    n_t : int
        Number of output time steps.
    initial_phi : callable, optional
        phi(x) at t=0. Default: Gaussian wave packet.
    initial_dphi_dt : callable, optional
        dphi/dt at t=0. Default: zero.
    hbar : float
        Reduced Planck constant.
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: x (1-d), t (1-d), phi (n_t, n_x), dispersion_relation.
    """
    if m < 0:
        raise ValueError("Mass must be >= 0.")

    x = np.linspace(x_range[0], x_range[1], n_x)
    dx = x[1] - x[0]

    if initial_phi is None:
        phi0 = np.exp(-x ** 2 / 2.0)
    else:
        phi0 = np.array([initial_phi(xi) for xi in x])

    if initial_dphi_dt is None:
        dphi0 = np.zeros(n_x)
    else:
        dphi0 = np.array([initial_dphi_dt(xi) for xi in x])

    mass_term = (m * c / hbar) ** 2

    def rhs(t, y):
        phi = y[:n_x]
        pi = y[n_x:]
        d2phi = np.zeros(n_x)
        d2phi[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx ** 2
        d2phi[0] = d2phi[1]
        d2phi[-1] = d2phi[-2]
        dphi_dt = pi
        dpi_dt = c ** 2 * (d2phi - mass_term * phi)
        return np.concatenate([dphi_dt, dpi_dt])

    y0 = np.concatenate([phi0, dphi0])
    t_eval = np.linspace(t_span[0], t_span[1], n_t)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="RK45")

    phi_out = sol.y[:n_x, :].T

    k = np.fft.fftfreq(n_x, d=dx) * 2 * np.pi
    omega = c * np.sqrt(k ** 2 + mass_term)

    return {
        "x": x,
        "t": sol.t,
        "phi": phi_out,
        "dispersion_relation": {"k": k, "omega": omega},
    }
