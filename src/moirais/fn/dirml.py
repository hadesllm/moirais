# moirais.fn — function file (hadesllm/moirais)
"""Directional spatial model (Rabinowitz-Macdonald)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def directional_model(ideal_points, alternatives) -> DescriptiveResult:
    """Out of chaos, comes order. — Friedrich Nietzsche"""
    X = np.asarray(ideal_points, dtype=float)
    Z = np.asarray(alternatives, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    utility = X @ Z.T
    predicted = np.argmax(utility, axis=1)

    return DescriptiveResult(
        name="directional_model",
        value={"utility": utility, "predicted_choice": predicted},
        extra={"n_voters": X.shape[0], "n_alternatives": Z.shape[0]},
    )


dirml = directional_model


def cheatsheet() -> str:
    return "directional_model({}) -> Directional spatial voting model."
