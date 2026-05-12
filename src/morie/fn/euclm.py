# morie.fn -- function file (hadesllm/morie)
"""Euclidean spatial model (utility = -distance^2)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def euclidean_model(ideal_points, alternatives, *, beta: float = 1.0) -> DescriptiveResult:
    """The happiness of your life depends upon the quality of your thoughts. -- Marcus Aurelius"""
    X = np.asarray(ideal_points, dtype=float)
    Z = np.asarray(alternatives, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    diff = X[:, None, :] - Z[None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    utility = -beta * dist_sq
    predicted = np.argmax(utility, axis=1)

    return DescriptiveResult(
        name="euclidean_model",
        value={"utility": utility, "predicted_choice": predicted},
        extra={"n_voters": X.shape[0], "n_alternatives": Z.shape[0], "beta": beta},
    )


euclm = euclidean_model


def cheatsheet() -> str:
    return "euclidean_model({}) -> Euclidean spatial voting model."
