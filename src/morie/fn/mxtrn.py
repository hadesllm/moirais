# morie.fn -- function file (hadesllm/morie)
"""Matrix trace, determinant, rank, condition number, and norms."""

import numpy as np

from ._containers import DescriptiveResult
def matrix_trace_norm(A, **kwargs) -> DescriptiveResult:
    """
    Compute matrix trace, determinant, rank, condition number, and
    Frobenius / spectral norms.

    :param A: (d, d) square matrix.
    :return: DescriptiveResult with all matrix properties.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    d = A.shape[0]
    tr = float(np.trace(A))
    det = float(np.linalg.det(A))
    rank = int(np.linalg.matrix_rank(A))
    cond = float(np.linalg.cond(A))
    frob = float(np.linalg.norm(A, "fro"))
    spectral = float(np.linalg.norm(A, 2))
    return DescriptiveResult(
        name="matrix_trace_norm",
        value=tr,
        extra={
            "trace": tr,
            "determinant": det,
            "rank": rank,
            "condition_number": cond,
            "frobenius_norm": frob,
            "spectral_norm": spectral,
            "dim": d,
        },
    )


mxtrn = matrix_trace_norm


def cheatsheet() -> str:
    return "matrix_trace_norm({}) -> Matrix trace, determinant, rank, condition number, and norms"
