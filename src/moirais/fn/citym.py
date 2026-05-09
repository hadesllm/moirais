# moirais.fn — function file (hadesllm/moirais)
"""City-block (Manhattan / L1) spatial model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cityblock_model(ideal_points, alternatives, *, beta: float = 1.0) -> DescriptiveResult:
    """City-block spatial voting model: U = -beta * sum|x_i - z_j|.

    L1 (Manhattan) distance assumes dimensions are separable and
    independently evaluated by voters.

    :param ideal_points: (n_voters x n_dims) voter ideal points.
    :param alternatives: (n_alternatives x n_dims) alternative positions.
    :param beta: Spatial weight parameter.
    :return: DescriptiveResult with utility matrix and predicted choices.

    References
    ----------
    Armstrong (2014), Ch 4.

    .. epigraph:: "These aren't the droids you'The whole is greater than the sum of its parts. — Aristotle'
    """
    X = np.asarray(ideal_points, dtype=float)
    Z = np.asarray(alternatives, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    diff = X[:, None, :] - Z[None, :, :]
    dist_l1 = np.abs(diff).sum(axis=-1)
    utility = -beta * dist_l1
    predicted = np.argmax(utility, axis=1)

    return DescriptiveResult(
        name="cityblock_model",
        value={"utility": utility, "predicted_choice": predicted},
        extra={"n_voters": X.shape[0], "n_alternatives": Z.shape[0], "beta": beta},
    )


citym = cityblock_model


def cheatsheet() -> str:
    return "cityblock_model({}) -> City-block (Manhattan) spatial model."
