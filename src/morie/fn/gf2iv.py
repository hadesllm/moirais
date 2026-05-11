# morie.fn — function file (hadesllm/morie)
"""GF(2) matrix inverse."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gf2_matrix_inv(a: np.ndarray) -> DescriptiveResult:
    """Compute the inverse of a matrix over GF(2).

    :param a: Square binary matrix.
    :return: DescriptiveResult with inverse matrix (None if singular).
    """
    from morie.crypto._gf2m import gf2_matrix_inv as _inv

    A = np.asarray(a, dtype=np.uint8)
    raw = _inv(A.tolist())
    success = raw is not None
    result = np.asarray(raw, dtype=np.uint8) if success else None
    return DescriptiveResult(
        name="gf2_matrix_inv",
        value=1.0 if success else 0.0,
        extra={
            "inverse": result,
            "invertible": success,
            "shape": list(A.shape),
        },
    )


gf2iv = gf2_matrix_inv


def cheatsheet() -> str:
    return "gf2_matrix_inv({}) -> GF(2) matrix inverse."
