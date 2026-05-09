"""Ordered Logit Spatial Model."""

import numpy as np

from ._containers import SpatialResult


def svord2(voter, candidates, *, beta=1.0):
    """Ordered Logit Spatial Model.

    Returns
    -------
    SpatialResult
    """
    voter = np.asarray(voter, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    dists = np.linalg.norm(candidates - voter, axis=1)
    ranks = dists.argsort().argsort()
    utils = -beta * dists
    exp_u = np.exp(utils - utils.max())
    probs = exp_u / exp_u.sum()
    stat = float(probs[ranks == 0][0]) if len(probs[ranks == 0]) > 0 else float(probs[0])
    _extra = {"probabilities": probs.tolist()}

    return SpatialResult(
        name="Ordered Logit Spatial Model",
        statistic=float(stat),
        extra=_extra,
    )


svord2 = svord2  # alias


def cheatsheet() -> str:
    return "svord2({}) -> Ordered Logit Spatial Model."
