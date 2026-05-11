"""Spatial voting utility computation (general)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spatial_utility(
    ideal_points,
    alternatives,
    *,
    model: str = "quadratic",
    beta: float = 1.0,
    w: float = 0.5,
) -> DescriptiveResult:
    """General spatial voting utility computation.

    Supports quadratic, Gaussian, linear, and mixed (proximity+directional).

    :param ideal_points: (n_voters x n_dims) voter ideal points.
    :param alternatives: (n_alternatives x n_dims) alternative positions.
    :param model: One of 'quadratic', 'gaussian', 'linear', 'mixed'.
    :param beta: Spatial weight / precision.
    :param w: Mixing weight for 'mixed' model (0=pure directional, 1=pure proximity).
    :return: DescriptiveResult with utility matrix and predicted choices.

    References
    ----------
    Armstrong (2014), Ch 4-5.

    .. epigraph:: 'Errors using inadequate data are much less than those using no data at all. — Charles Babbage'
    """
    X = np.asarray(ideal_points, dtype=float)
    Z = np.asarray(alternatives, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    diff = X[:, None, :] - Z[None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)

    if model == "quadratic":
        utility = -beta * dist_sq
    elif model == "gaussian":
        utility = np.exp(-0.5 * beta * dist_sq)
    elif model == "linear":
        utility = -beta * np.sqrt(dist_sq + 1e-14)
    elif model == "mixed":
        proximity = -beta * dist_sq
        directional = X @ Z.T
        utility = w * proximity + (1 - w) * directional
    else:
        raise ValueError(f"Unknown model: {model}. Use quadratic/gaussian/linear/mixed.")

    predicted = np.argmax(utility, axis=1)
    return DescriptiveResult(
        name="spatial_utility",
        value={"utility": utility, "predicted_choice": predicted},
        extra={"model": model, "beta": beta},
    )


spvut = spatial_utility


def cheatsheet() -> str:
    return "spatial_utility({}) -> General spatial voting utility computation."
