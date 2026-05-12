# morie.fn -- function file (hadesllm/morie)
"""Moore-Penrose pseudoinverse."""

import numpy as np

from ._containers import DescriptiveResult


def pseudoinverse(matrix: np.ndarray) -> DescriptiveResult:
    """
    Compute the Moore-Penrose pseudoinverse.

    :param matrix: (m, n) matrix.
    :return: DescriptiveResult with pseudoinverse in extra.

    References
    ----------
    Penrose R (1955). A generalized inverse for matrices. Proceedings
    of the Cambridge Philosophical Society, 51(3), 406-413.
    """
    A = np.asarray(matrix, dtype=np.float64)
    A_pinv = np.linalg.pinv(A)
    rank = int(np.linalg.matrix_rank(A))
    return DescriptiveResult(
        name="pseudoinverse",
        value=float(rank),
        extra={"pinv": A_pinv, "rank": rank, "shape": A.shape, "pinv_shape": A_pinv.shape},
    )


pinv = pseudoinverse


def cheatsheet() -> str:
    return "pseudoinverse({}) -> Moore-Penrose pseudoinverse."
