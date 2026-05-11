# morie.fn — function file (hadesllm/morie)
"""GF(2) matrix multiplication."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gf2_matrix_mul(a: np.ndarray, b: np.ndarray) -> DescriptiveResult:
    """Multiply two matrices over GF(2).

    :param a: First binary matrix.
    :param b: Second binary matrix.
    :return: DescriptiveResult with product matrix.
    """
    from morie.crypto._gf2m import gf2_matrix_mul as _mul

    A = np.asarray(a, dtype=np.uint8)
    B = np.asarray(b, dtype=np.uint8)
    result = np.asarray(_mul(A.tolist(), B.tolist()), dtype=np.uint8)
    return DescriptiveResult(
        name="gf2_matrix_mul",
        value=float(np.sum(result)),
        extra={"result": result, "shape": list(result.shape)},
    )


gf2ml = gf2_matrix_mul


def cheatsheet() -> str:
    return "gf2_matrix_mul({}) -> GF(2) matrix multiplication."
