"""1-D wave equation solver. 'Shockwave calculates.' -- Shockwave"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wave_equation_1d(
    n_x: int = 200,
    n_t: int = 500,
    *,
    c: float = 1.0,
    dx: float = 0.01,
    dt: float | None = None,
    ic_func=None,
) -> DescriptiveResult:
    """Solve the 1-D wave equation using finite differences (FTCS explicit).

    u_tt = c^2 * u_xx on [0, L] with fixed boundary conditions u(0)=u(L)=0.

    Parameters
    ----------
    n_x : int
        Number of spatial grid points.
    n_t : int
        Number of time steps.
    c : float
        Wave speed.
    dx : float
        Spatial step size.
    dt : float, optional
        Time step. Defaults to CFL-stable: 0.9 * dx / c.
    ic_func : callable, optional
        Initial condition u(x, 0). Defaults to sin(pi * x / L).

    Returns
    -------
    DescriptiveResult
        With ``value`` = solution array (n_t x n_x) and
        ``extra`` containing CFL number.
    """
    if dt is None:
        dt = 0.9 * dx / c
    cfl = c * dt / dx
    if cfl > 1.0:
        raise ValueError(f"CFL={cfl:.3f} > 1: unstable. Reduce dt or increase dx.")

    L = dx * (n_x - 1)
    x = np.linspace(0, L, n_x)

    u = np.zeros((n_t, n_x))
    if ic_func is None:
        u[0] = np.sin(np.pi * x / L)
    else:
        u[0] = np.array([ic_func(xi) for xi in x])
    u[0, 0] = u[0, -1] = 0.0

    r2 = cfl**2
    u[1, 1:-1] = u[0, 1:-1] + 0.5 * r2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])

    for t in range(1, n_t - 1):
        u[t + 1, 1:-1] = 2 * u[t, 1:-1] - u[t - 1, 1:-1] + r2 * (u[t, 2:] - 2 * u[t, 1:-1] + u[t, :-2])

    return DescriptiveResult(
        name="wave_equation_1d",
        value=u,
        extra={"cfl": cfl, "c": c, "dx": dx, "dt": dt, "n_x": n_x, "n_t": n_t, "L": L},
    )


shkwv = wave_equation_1d


def cheatsheet() -> str:
    return "wave_equation_1d({}) -> 1-D wave equation solver. 'Shockwave calculates.' -- Shockwa"
