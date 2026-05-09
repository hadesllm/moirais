# moirais.fn — function file (hadesllm/moirais)
"""1D heat diffusion solver. 'Your focus determines your reality.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def heat_diffusion(
    T0: np.ndarray,
    alpha: float = 0.01,
    dx: float = 0.1,
    dt: float = 0.01,
    n_steps: int = 100,
) -> DescriptiveResult:
    """
    Solve the 1D heat equation via explicit finite differences.

    .. math::

        \\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}

    Uses forward Euler with Dirichlet boundary conditions (T fixed at
    endpoints).

    :param T0: Initial temperature profile (1D array).
    :param alpha: Thermal diffusivity (m^2/s). Default 0.01.
    :param dx: Spatial step (m). Default 0.1.
    :param dt: Time step (s). Default 0.01.
    :param n_steps: Number of time steps. Default 100.
    :return: DescriptiveResult with final temperature and stability info.
    :raises ValueError: If CFL condition is violated (r > 0.5).

    References
    ----------
    Crank, J. (1975). *The Mathematics of Diffusion*. 2nd ed.
    Oxford University Press.
    """
    T0 = np.asarray(T0, dtype=np.float64)
    if T0.ndim != 1 or len(T0) < 3:
        raise ValueError("T0 must be a 1D array with at least 3 points.")

    r = alpha * dt / (dx**2)
    if r > 0.5:
        raise ValueError(f"CFL condition violated: r = {r:.4f} > 0.5. Reduce dt or increase dx.")

    n_x = len(T0)
    T = T0.copy()
    history = [T.copy()]

    for _ in range(n_steps):
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
        T = T_new
        history.append(T.copy())

    return DescriptiveResult(
        name="1D Heat Diffusion",
        value=float(T.mean()),
        extra={
            "T_final": T,
            "T_initial": T0,
            "history": np.array(history),
            "r_stability": float(r),
            "n_steps": n_steps,
            "alpha": alpha,
        },
    )


short = heat_diffusion


def cheatsheet() -> str:
    return "heat_diffusion({}) -> 1D heat diffusion solver. 'Your focus determines your realit"
