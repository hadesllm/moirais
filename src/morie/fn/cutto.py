# morie.fn — function file (hadesllm/morie)
"""Cutting plane / separating hyperplane for spatial voting."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cutting_plane(z1, z2, *, ideal_points=None) -> DescriptiveResult:
    """Compute the cutting plane between two alternatives in spatial voting.

    The cutting plane is the hyperplane equidistant from z1 and z2.
    Under Euclidean preferences, voters on the z1 side prefer z1.

    Normal vector: n = z1 - z2
    Midpoint: m = (z1 + z2) / 2
    Plane: n . (x - m) = 0

    :param z1: Position of alternative 1.
    :param z2: Position of alternative 2.
    :param ideal_points: Optional (n x d) voter positions to classify.
    :return: DescriptiveResult with normal vector, midpoint, and classifications.

    References
    ----------
    Armstrong (2014), Ch 5-6.

    .. epigraph:: "Luminous beings are we, not this crude matter.", Star Wars
    """
    a = np.asarray(z1, dtype=float).ravel()
    b = np.asarray(z2, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError("z1 and z2 must have the same dimensionality.")

    normal = a - b
    midpoint = (a + b) / 2.0

    result = {
        "normal": normal,
        "midpoint": midpoint,
        "norm_length": float(np.linalg.norm(normal)),
    }

    if ideal_points is not None:
        X = np.asarray(ideal_points, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, len(a))
        projections = (X - midpoint) @ normal
        classifications = (projections > 0).astype(int)
        result["classifications"] = classifications
        result["n_prefer_z1"] = int(classifications.sum())
        result["n_prefer_z2"] = int(len(classifications) - classifications.sum())

    return DescriptiveResult(
        name="cutting_plane",
        value=result,
        extra={"n_dims": len(a)},
    )


cutto = cutting_plane


def cheatsheet() -> str:
    return "cutting_plane({}) -> Cutting plane / separating hyperplane."
