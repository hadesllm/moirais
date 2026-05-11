"""Kruskal stress-1. 'Texas Smash!' -- All Might, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def stress1_measure(D_obs, D_model):
    """Kruskal stress-1: sqrt(sum((d_obs - d_model)^2) / sum(d_obs^2)).

    Parameters
    ----------
    D_obs : array-like
        Observed distance matrix.
    D_model : array-like
        Model distance matrix.

    Returns
    -------
    DescriptiveResult
        value = stress-1 (float).
    """
    import numpy as np

    D_obs = np.asarray(D_obs, dtype=float)
    D_model = np.asarray(D_model, dtype=float)
    triu = np.triu_indices(D_obs.shape[0], k=1)
    obs = D_obs[triu]
    mod = D_model[triu]
    num = np.sum((obs - mod) ** 2)
    denom = np.sum(obs**2)
    s1 = float(np.sqrt(num / denom)) if denom > 0 else 0.0
    return DescriptiveResult(
        name="stress1_measure", value=s1, extra={"numerator": float(num), "denominator": float(denom)}
    )


strs1 = stress1_measure


def cheatsheet() -> str:
    return "stress1_measure({}) -> Kruskal stress-1. 'Texas Smash!' -- All Might, My Hero Acade"
