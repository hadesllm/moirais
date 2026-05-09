"""Torus bundle classification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_bundle(monodromy: list | np.ndarray = None) -> DescriptiveResult:
    """Classify a torus bundle over S^1 by its monodromy matrix.

    A torus bundle over the circle is determined (up to isomorphism)
    by a conjugacy class in SL(2, Z). The trace classifies the geometry:
    |tr| < 2 -> elliptic (Euclidean), |tr| = 2 -> parabolic (Nil/Euclidean),
    |tr| > 2 -> hyperbolic (Sol).

    :param monodromy: 2x2 integer matrix in SL(2,Z). Default: identity.
    :return: DescriptiveResult with geometry classification.
    """
    if monodromy is None:
        monodromy = [[1, 0], [0, 1]]
    M = np.asarray(monodromy, dtype=int)
    if M.shape != (2, 2):
        raise ValueError(f"Monodromy must be 2x2, got {M.shape}.")
    det = int(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    if det != 1:
        raise ValueError(f"Determinant must be 1, got {det}.")
    tr = int(M[0, 0] + M[1, 1])
    abs_tr = abs(tr)
    if abs_tr < 2:
        geometry = "Euclidean"
    elif abs_tr == 2:
        is_id = np.array_equal(M, np.eye(2, dtype=int))
        is_neg_id = np.array_equal(M, -np.eye(2, dtype=int))
        geometry = "Euclidean" if (is_id or is_neg_id) else "Nil"
    else:
        geometry = "Sol"
    return DescriptiveResult(
        name="torus_bundle",
        value=float(tr),
        extra={
            "monodromy": M.tolist(),
            "trace": tr,
            "geometry": geometry,
            "determinant": det,
        },
    )


def cheatsheet() -> str:
    return "torus_bundle(monodromy) -> torus bundle geometry classification"
