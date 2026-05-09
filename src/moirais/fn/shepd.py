"""Shepard diagram data. 'Plus Ultra!' -- All Might, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def shepard_diagram(D_obs, D_model):
    """Generate data for a Shepard diagram (observed vs fitted distances).

    Parameters
    ----------
    D_obs : array-like
        Observed distance matrix.
    D_model : array-like
        Model distance matrix.

    Returns
    -------
    DescriptiveResult
        value = correlation, extra has observed and fitted arrays.
    """
    import numpy as np

    D_obs = np.asarray(D_obs, dtype=float)
    D_model = np.asarray(D_model, dtype=float)
    n = D_obs.shape[0]
    triu = np.triu_indices(n, k=1)
    obs = D_obs[triu]
    mod = D_model[triu]
    r = float(np.corrcoef(obs, mod)[0, 1]) if len(obs) > 1 else 0.0
    return DescriptiveResult(
        name="shepard_diagram",
        value=r,
        extra={"observed": obs, "fitted": mod, "n_pairs": len(obs)},
    )


shepd = shepard_diagram


def cheatsheet() -> str:
    return "shepard_diagram({}) -> Shepard diagram data. 'Plus Ultra!' -- All Might, My Hero Ac"
