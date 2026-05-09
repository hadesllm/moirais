# moirais.fn — function file (hadesllm/moirais)
"""Particle Swarm Optimization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def particle_swarm(
    f: Callable,
    bounds: list[tuple[float, float]],
    *,
    n_particles: int = 30,
    maxiter: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int = 42,
) -> DescriptiveResult:
    """Particle Swarm Optimization (PSO).

    Global metaheuristic optimizer for continuous functions.

    Parameters
    ----------
    f : callable
        Objective function f(x) -> scalar.
    bounds : list of (lo, hi)
        Search bounds per dimension.
    n_particles : int
        Swarm size.
    maxiter : int
        Maximum iterations.
    w : float
        Inertia weight.
    c1 : float
        Cognitive (personal best) coefficient.
    c2 : float
        Social (global best) coefficient.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the best objective found; ``extra`` has x_best.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pos = rng.uniform(lo, hi, (n_particles, d))
    vel = rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1, (n_particles, d))
    pbest = pos.copy()
    pbest_val = np.array([f(pos[i]) for i in range(n_particles)])
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    for _ in range(maxiter):
        r1 = rng.uniform(0, 1, (n_particles, d))
        r2 = rng.uniform(0, 1, (n_particles, d))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        for i in range(n_particles):
            val = f(pos[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = pos[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest = pos[i].copy()
    return DescriptiveResult(
        name="PSO",
        value=float(gbest_val),
        extra={"x": gbest},
    )


pswrm = particle_swarm
