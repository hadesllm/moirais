# morie.fn — function file (hadesllm/morie)
"""System failure cascade / reliability. 'We are the Doom Patrol.' -- The Chief"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def failure_cascade(
    component_reliabilities: np.ndarray | list[float],
    *,
    topology: str = "series",
    n_sim: int = 10000,
    seed: int | None = None,
) -> DescriptiveResult:
    """System reliability via Monte Carlo failure cascade simulation.

    Simulates whether each component survives (Bernoulli with given
    reliability) and evaluates system-level reliability under series,
    parallel, or k-of-n topology.

    Parameters
    ----------
    component_reliabilities : array
        Per-component reliability probabilities (0 to 1).
    topology : str
        ``'series'`` (all must work), ``'parallel'`` (any works), or ``'k_of_n'``.
    n_sim : int
        Monte Carlo replications.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` = estimated system reliability.
    """
    r = np.asarray(component_reliabilities, dtype=float).ravel()
    if len(r) < 1:
        raise ValueError("Need at least 1 component")
    if np.any(r < 0) or np.any(r > 1):
        raise ValueError("Reliabilities must be in [0, 1]")
    n_comp = len(r)
    rng = np.random.default_rng(seed)
    draws = rng.random((n_sim, n_comp))
    alive = draws < r
    if topology == "series":
        system_ok = np.all(alive, axis=1)
        exact = float(np.prod(r))
    elif topology == "parallel":
        system_ok = np.any(alive, axis=1)
        exact = float(1 - np.prod(1 - r))
    elif topology == "k_of_n":
        k = max(1, n_comp // 2 + 1)
        system_ok = np.sum(alive, axis=1) >= k
        exact = None
    else:
        raise ValueError(f"Unknown topology: {topology}")
    sim_rel = float(np.mean(system_ok))
    se = float(np.sqrt(sim_rel * (1 - sim_rel) / n_sim))
    return DescriptiveResult(
        name=f"System reliability ({topology})",
        value=sim_rel,
        extra={
            "n_components": n_comp,
            "topology": topology,
            "n_sim": n_sim,
            "se": se,
            "exact_reliability": exact,
            "component_reliabilities": r.tolist(),
            "weakest_component": int(np.argmin(r)),
            "min_reliability": float(np.min(r)),
        },
    )


doom = failure_cascade


def cheatsheet() -> str:
    return "failure_cascade({}) -> System failure cascade / reliability. 'We are the Doom Patro"
