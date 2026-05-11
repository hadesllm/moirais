# morie.fn — function file (hadesllm/morie)
"""Geodesic equation solver (Runge-Kutta on curved spacetime)."""

__all__ = ["geods"]

import numpy as np
from scipy.integrate import solve_ivp


def geods(
    metric_func,
    x0: np.ndarray,
    u0: np.ndarray,
    tau_span: tuple = (0.0, 10.0),
    n_points: int = 500,
    h: float = 1e-5,
) -> dict:
    """
    Solve the geodesic equation via numerical integration.

    .. math::

        \\frac{d^2 x^\\mu}{d\\tau^2}
        + \\Gamma^\\mu_{\\alpha\\beta}
          \\frac{dx^\\alpha}{d\\tau}\\frac{dx^\\beta}{d\\tau} = 0

    Christoffel symbols are computed numerically from the metric.

    Parameters
    ----------
    metric_func : callable
        metric_func(x) -> (4,4) ndarray, the metric at position x.
    x0 : np.ndarray
        Initial 4-position [t, r, theta, phi] (or similar coords).
    u0 : np.ndarray
        Initial 4-velocity.
    tau_span : tuple
        (tau_start, tau_end) proper time range.
    n_points : int
        Number of output points.
    h : float
        Step size for numerical differentiation of metric.

    Returns
    -------
    dict
        Keys: tau (1-d), position (n,4), velocity (n,4).
    """
    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    if x0.shape != (4,) or u0.shape != (4,):
        raise ValueError("x0 and u0 must be length-4.")

    def christoffel(x):
        g = metric_func(x)
        ginv = np.linalg.inv(g)
        dg = np.zeros((4, 4, 4))
        for mu in range(4):
            dx = np.zeros(4)
            dx[mu] = h
            gp = metric_func(x + dx)
            gm = metric_func(x - dx)
            dg[mu] = (gp - gm) / (2.0 * h)
        G = np.zeros((4, 4, 4))
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    s = 0.0
                    for sig in range(4):
                        s += 0.5 * ginv[lam, sig] * (
                            dg[nu, sig, mu] + dg[mu, sig, nu] - dg[sig, mu, nu]
                        )
                    G[lam, mu, nu] = s
        return G

    def rhs(tau, y):
        x = y[:4]
        u = y[4:]
        G = christoffel(x)
        accel = np.zeros(4)
        for mu in range(4):
            s = 0.0
            for a in range(4):
                for b in range(4):
                    s += G[mu, a, b] * u[a] * u[b]
            accel[mu] = -s
        return np.concatenate([u, accel])

    y0 = np.concatenate([x0, u0])
    tau_eval = np.linspace(tau_span[0], tau_span[1], n_points)
    sol = solve_ivp(rhs, tau_span, y0, t_eval=tau_eval, method="RK45",
                    rtol=1e-10, atol=1e-12)

    return {
        "tau": sol.t,
        "position": sol.y[:4].T,
        "velocity": sol.y[4:].T,
    }
