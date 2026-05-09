"""Multinomial Logit Vote Choice."""

import numpy as np

from ._containers import SpatialResult


def svmnl2(voter, candidates, *, beta=1.0):
    """Multinomial Logit Vote Choice.

    Returns
    -------
    SpatialResult
    """
    voter = np.asarray(voter, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    utils = -beta * np.sum((candidates - voter) ** 2, axis=1)
    exp_u = np.exp(utils - utils.max())
    probs = exp_u / exp_u.sum()
    stat = float(probs.max())
    _extra = {"probabilities": probs.tolist(), "chosen": int(np.argmax(probs))}

    return SpatialResult(
        name="Multinomial Logit Vote Choice",
        statistic=float(stat),
        extra=_extra,
    )


svmnl2 = svmnl2  # alias


def cheatsheet() -> str:
    return "svmnl2({}) -> Multinomial Logit Vote Choice."
