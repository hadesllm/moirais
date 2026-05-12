# morie.fn -- function file (hadesllm/morie)
"""Moore-Penrose pseudoinverse."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Knowing others is intelligence; knowing yourself is true wisdom. -- Lao Tzu"


def pseudo_inverse(A, **kwargs) -> DescriptiveResult:
    """
    Compute the Moore-Penrose pseudoinverse A+.

    .. math::

        A^{+} = V S^{+} U^T

    where S+ is the diagonal matrix with reciprocals of non-zero
    singular values.

    :param A: (m, n) matrix.
    :return: DescriptiveResult with pseudoinverse shape and rank.

    References
    ----------
    Penrose R (1955). A generalized inverse for matrices.
    Mathematical Proceedings of the Cambridge Philosophical Society,
    51(3), 406-413.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    A_pinv = np.linalg.pinv(A)
    rank = int(np.linalg.matrix_rank(A))
    return DescriptiveResult(
        name="pseudo_inverse",
        value=float(rank),
        extra={
            "shape": list(A.shape),
            "pinv_shape": list(A_pinv.shape),
            "rank": rank,
            "frobenius_norm": float(np.linalg.norm(A_pinv, "fro")),
        },
    )


psinv = pseudo_inverse


def cheatsheet() -> str:
    return "pseudo_inverse({}) -> Moore-Penrose pseudoinverse."
