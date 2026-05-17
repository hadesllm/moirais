"""Minimise a function via particle swarm optimisation (PSO)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def swarm_optimize(
    objective,
    bounds: list[tuple[float, float]],
    *,
    n_particles: int = 30,
    n_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int | None = None,
) -> DescriptiveResult:
    """Minimise a function via particle swarm optimisation (PSO).

    Parameters
    ----------
    objective : callable
        Function f(x) -> float to minimise.  x is (d,) array.
    bounds : list of (lo, hi)
        Variable bounds per dimension.
    n_particles : int
        Swarm size.
    n_iter : int
        Number of iterations.
    w : float
        Inertia weight.
    c1, c2 : float
        Cognitive and social acceleration coefficients.
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``x_best``, ``f_best``, ``convergence`` (history).
    """
    d = len(bounds)
    if d < 1:
        raise ValueError("Need at least 1 dimension")

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    pos = rng.uniform(lo, hi, size=(n_particles, d))
    vel = rng.uniform(-(hi - lo), hi - lo, size=(n_particles, d))

    pbest_pos = pos.copy()
    pbest_val = np.array([objective(pos[i]) for i in range(n_particles)])
    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [float(gbest_val)]

    for _ in range(n_iter):
        r1 = rng.uniform(0, 1, (n_particles, d))
        r2 = rng.uniform(0, 1, (n_particles, d))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, lo, hi)

        for i in range(n_particles):
            val = objective(pos[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = pos[i].copy()

        history.append(float(gbest_val))

    return DescriptiveResult(
        name="swarm_optimize",
        value={
            "x_best": gbest_pos,
            "f_best": float(gbest_val),
            "convergence": np.array(history),
        },
        extra={"n_particles": n_particles, "n_iter": n_iter, "d": d},
    )


swaopt = swarm_optimize


def cheatsheet() -> str:
    return 'swarm_optimize({}) -> Particle swarm optimization.'
