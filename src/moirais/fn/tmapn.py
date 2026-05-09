"""Mapping class group action on the torus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_mapping_class(matrix: list | np.ndarray = None) -> DescriptiveResult:
    """Apply an SL(2,Z) mapping class group element to the torus.

    The mapping class group of T^2 is SL(2, Z). Given a 2x2 integer
    matrix with determinant 1, compute its action on the homology basis
    and classify the element (elliptic, parabolic, hyperbolic).

    :param matrix: 2x2 integer matrix in SL(2,Z). Default is identity.
    :return: DescriptiveResult with classification and trace.
    """
    if matrix is None:
        matrix = [[1, 0], [0, 1]]
    M = np.asarray(matrix, dtype=int)
    if M.shape != (2, 2):
        raise ValueError(f"Matrix must be 2x2, got {M.shape}.")
    det = int(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    if det != 1:
        raise ValueError(f"Matrix must have determinant 1, got {det}.")
    tr = int(M[0, 0] + M[1, 1])
    abs_tr = abs(tr)
    if abs_tr < 2:
        classification = "elliptic"
    elif abs_tr == 2:
        classification = "parabolic"
    else:
        classification = "hyperbolic"
    order = None
    if tr == 2 and np.array_equal(M, np.eye(2, dtype=int)):
        order = 1
    elif tr == -2 and np.array_equal(M, -np.eye(2, dtype=int)):
        order = 2
    elif tr == -1:
        order = 3
    elif tr == 0:
        order = 4
    elif tr == 1:
        order = 6
    return DescriptiveResult(
        name="torus_mapping_class",
        value=float(tr),
        extra={
            "matrix": M.tolist(),
            "trace": tr,
            "determinant": det,
            "classification": classification,
            "order": order,
        },
    )


def cheatsheet() -> str:
    return "torus_mapping_class(matrix) -> SL(2,Z) classification"
