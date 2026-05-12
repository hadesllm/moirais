"""1D wave equation solver. 'It is the mark of an educated mind to entertain a thought without accepting it. -- Aristotle'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wave_1d(
    u0: np.ndarray,
    c: float = 1.0,
    dx: float = 0.1,
    dt: float = 0.01,
    n_steps: int = 100,
) -> DescriptiveResult:
    r"""
    Solve the 1D wave equation via explicit finite differences.

    .. math::

        \\frac{\\partial^2 u}{\\partial t^2} = c^2
        \\frac{\\partial^2 u}{\\partial x^2}

    Uses a three-level leapfrog scheme with fixed boundary conditions.

    :param u0: Initial displacement profile (1D array).
    :param c: Wave speed. Default 1.0.
    :param dx: Spatial step. Default 0.1.
    :param dt: Time step. Default 0.01.
    :param n_steps: Number of time steps. Default 100.
    :return: DescriptiveResult with final displacement and CFL number.
    :raises ValueError: If CFL condition violated (courant > 1).

    References
    ----------
    Strikwerda, J. C. (2004). *Finite Difference Schemes and Partial
    Differential Equations*. 2nd ed. SIAM.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    if u0.ndim != 1 or len(u0) < 3:
        raise ValueError("u0 must be a 1D array with at least 3 points.")

    courant = c * dt / dx
    if courant > 1.0:
        raise ValueError(f"CFL condition violated: C = {courant:.4f} > 1. Reduce dt or increase dx.")

    c2 = courant**2
    n_x = len(u0)

    u_prev = u0.copy()
    u_curr = u0.copy()
    u_curr[1:-1] = u0[1:-1] + 0.5 * c2 * (u0[2:] - 2 * u0[1:-1] + u0[:-2])

    history = [u_prev.copy(), u_curr.copy()]

    for _ in range(n_steps - 1):
        u_next = np.zeros(n_x)
        u_next[1:-1] = 2 * u_curr[1:-1] - u_prev[1:-1] + c2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2])
        u_prev = u_curr
        u_curr = u_next
        history.append(u_curr.copy())

    return DescriptiveResult(
        name="1D Wave Equation",
        value=float(np.max(np.abs(u_curr))),
        extra={
            "u_final": u_curr,
            "u_initial": u0,
            "history": np.array(history),
            "courant_number": float(courant),
            "wave_speed": c,
            "n_steps": n_steps,
        },
    )


short = wave_1d


def cheatsheet() -> str:
    return "wave_1d({}) -> 1D wave equation solver. 'It is the mark of an educated mind to entertain a thought without accepting it. -- Aristotle' "
