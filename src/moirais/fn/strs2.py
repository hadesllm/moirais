"""Stress-2 measure. 'United States of Smash!' -- All Might, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def stress2_measure(D_obs, D_model):
    """Stress-2: sqrt(sum((d_obs - d_model)^2) / sum((d_obs - d_obs_mean)^2)).

    Parameters
    ----------
    D_obs : array-like
        Observed distance matrix.
    D_model : array-like
        Model distance matrix.

    Returns
    -------
    DescriptiveResult
        value = stress-2 (float).
    """
    import numpy as np

    D_obs = np.asarray(D_obs, dtype=float)
    D_model = np.asarray(D_model, dtype=float)
    triu = np.triu_indices(D_obs.shape[0], k=1)
    obs = D_obs[triu]
    mod = D_model[triu]
    num = np.sum((obs - mod) ** 2)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    s2 = float(np.sqrt(num / denom)) if denom > 0 else 0.0
    return DescriptiveResult(
        name="stress2_measure", value=s2, extra={"numerator": float(num), "denominator": float(denom)}
    )


strs2 = stress2_measure


def cheatsheet() -> str:
    return "stress2_measure({}) -> Stress-2 measure. 'United States of Smash!' -- All Might, My"
