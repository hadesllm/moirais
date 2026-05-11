"""Mixed Logit Spatial Model."""

import numpy as np

from ._containers import SpatialResult


def svmxl2(voter, candidates, *, beta=1.0, n_draws=100):
    """Mixed Logit Spatial Model.

    Returns
    -------
    SpatialResult
    """
    voter = np.asarray(voter, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    rng = np.random.default_rng(0)
    betas = rng.normal(beta, 0.1, n_draws)
    probs_all = []
    for b in betas:
        utils = -b * np.sum((candidates - voter) ** 2, axis=1)
        exp_u = np.exp(utils - utils.max())
        probs_all.append(exp_u / exp_u.sum())
    mean_probs = np.mean(probs_all, axis=0)
    stat = float(mean_probs.max())
    _extra = {"mean_probs": mean_probs.tolist(), "n_draws": n_draws}

    return SpatialResult(
        name="Mixed Logit Spatial Model",
        statistic=float(stat),
        extra=_extra,
    )


svmxl2 = svmxl2  # alias


def cheatsheet() -> str:
    return "svmxl2({}) -> Mixed Logit Spatial Model."
