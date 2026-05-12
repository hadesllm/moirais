# morie.fn -- function file (hadesllm/morie)
"""GF(2) matrix addition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gf2_matrix_add(a: np.ndarray, b: np.ndarray) -> DescriptiveResult:
    """Add two matrices over GF(2) (element-wise XOR).

    :param a: First binary matrix.
    :param b: Second binary matrix.
    :return: DescriptiveResult with sum matrix.
    """
    from morie.crypto._gf2m import gf2_matrix_add as _add

    A = np.asarray(a, dtype=np.uint8)
    B = np.asarray(b, dtype=np.uint8)
    result = np.asarray(_add(A.tolist(), B.tolist()), dtype=np.uint8)
    return DescriptiveResult(
        name="gf2_matrix_add",
        value=float(np.sum(result)),
        extra={"result": result, "shape": list(result.shape)},
    )


gf2ad = gf2_matrix_add


def cheatsheet() -> str:
    return "gf2_matrix_add({}) -> GF(2) matrix addition."
