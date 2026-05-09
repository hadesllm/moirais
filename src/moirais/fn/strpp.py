"""Stress per point. 'Domain Expansion!' -- Gojo, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def stress_per_point(D_obs, D_model):
    """Compute stress contribution per object.

    Parameters
    ----------
    D_obs : array-like
        Observed distance matrix (n x n).
    D_model : array-like
        Model distance matrix (n x n).

    Returns
    -------
    DescriptiveResult
        value = stress per point array (length n).
    """
    import numpy as np

    D_obs = np.asarray(D_obs, dtype=float)
    D_model = np.asarray(D_model, dtype=float)
    n = D_obs.shape[0]
    spp = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                spp[i] += (D_obs[i, j] - D_model[i, j]) ** 2
    total = np.sum(spp)
    if total > 0:
        spp = spp / total
    return DescriptiveResult(name="stress_per_point", value=spp, extra={"n": n, "total_stress": float(total)})


strpp = stress_per_point


def cheatsheet() -> str:
    return "stress_per_point({}) -> Stress per point. 'Domain Expansion!' -- Gojo, Jujutsu Kaise"
