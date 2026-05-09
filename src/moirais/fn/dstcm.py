# moirais.fn — function file (hadesllm/moirais)
"""Distance comparison stats. 'Blackwhip.' -- Deku, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def distance_comparison(D_obs, D_model):
    """Compare observed and model distances: correlation and RMSE.

    Parameters
    ----------
    D_obs : array-like
        Observed distance matrix.
    D_model : array-like
        Model distance matrix.

    Returns
    -------
    DescriptiveResult
        value = Pearson r, extra has rmse and n_pairs.
    """
    import numpy as np

    D_obs = np.asarray(D_obs, dtype=float)
    D_model = np.asarray(D_model, dtype=float)
    triu = np.triu_indices(D_obs.shape[0], k=1)
    obs = D_obs[triu]
    mod = D_model[triu]
    r = float(np.corrcoef(obs, mod)[0, 1]) if len(obs) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((obs - mod) ** 2)))
    return DescriptiveResult(name="distance_comparison", value=r, extra={"rmse": rmse, "r": r, "n_pairs": len(obs)})


dstcm = distance_comparison


def cheatsheet() -> str:
    return "distance_comparison({}) -> Distance comparison stats. 'Blackwhip.' -- Deku, My Hero Aca"
