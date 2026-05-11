# morie.fn — function file (hadesllm/morie)
"""Disparity fit via isotonic regression. 'Detroit Smash!' -- All Might, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def disparity_fit(D_obs, D_model):
    """Compute disparities from observed and model distances using isotonic regression.

    Parameters
    ----------
    D_obs : array-like
        Observed distance matrix.
    D_model : array-like
        Model (fitted) distance matrix.

    Returns
    -------
    DescriptiveResult
        value = disparity vector, extra has indices.
    """
    import numpy as np

    D_obs = np.asarray(D_obs, dtype=float)
    D_model = np.asarray(D_model, dtype=float)
    n = D_obs.shape[0]
    triu = np.triu_indices(n, k=1)
    obs_flat = D_obs[triu]
    mod_flat = D_model[triu]

    order = np.argsort(obs_flat)
    obs_sorted = obs_flat[order]
    mod_sorted = mod_flat[order]

    from morie.fn.isorg import isotonic_regression

    iso_res = isotonic_regression(mod_sorted)
    disparities = iso_res.value

    disp_full = np.empty_like(obs_flat)
    disp_full[order] = disparities
    return DescriptiveResult(name="disparity_fit", value=disp_full, extra={"n_pairs": len(disp_full)})


dispr = disparity_fit


def cheatsheet() -> str:
    return "disparity_fit({}) -> Disparity fit via isotonic regression. 'Detroit Smash!' -- A"
