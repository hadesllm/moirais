"""Simulation envelope for point pattern statistics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulation_envelope(points, window, stat_fn, n_sim=99, seed=None):
    """Compute simulation envelopes for a point pattern statistic under CSR.

    .. epigraph:: "Hmm." -- Geralt, The Witcher

    Parameters
    ----------
    points : array_like
        Observed point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    stat_fn : callable
        ``stat_fn(points) -> array`` returning statistic values.
    n_sim : int
        Number of CSR simulations.
    seed : int, optional
        Random seed.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window

    observed = np.asarray(stat_fn(pts))
    rng = np.random.default_rng(seed)

    sim_results = []
    for _ in range(n_sim):
        sx = rng.uniform(xmin, xmax, n)
        sy = rng.uniform(ymin, ymax, n)
        sim_pts = np.column_stack([sx, sy])
        sim_results.append(np.asarray(stat_fn(sim_pts)))

    sim_arr = np.array(sim_results)
    lo = np.percentile(sim_arr, 2.5, axis=0)
    hi = np.percentile(sim_arr, 97.5, axis=0)
    mean_env = np.mean(sim_arr, axis=0)

    outside = np.any((observed < lo) | (observed > hi))

    return DescriptiveResult(
        name="simulation_envelope",
        value=float(outside),
        extra={
            "observed": observed.tolist(),
            "envelope_lo": lo.tolist(),
            "envelope_hi": hi.tolist(),
            "envelope_mean": mean_env.tolist(),
            "significant": bool(outside),
            "n_sim": n_sim,
        },
    )


sgenv = simulation_envelope


def cheatsheet() -> str:
    return "simulation_envelope({}) -> Simulation envelope for point pattern statistics."
