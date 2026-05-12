"""State feedback gain via pole placement."""

import numpy as np

from ._containers import DescriptiveResult
def state_feedback(A, B, poles, **kwargs) -> DescriptiveResult:
    r"""
    Compute state feedback gain matrix via Ackermann's pole placement.

    For a controllable system :math:`\\dot{x} = Ax + Bu`, find gain :math:`K`
    such that :math:`\\dot{x} = (A - BK)x` has the desired closed-loop poles.

    Ackermann's formula:

    .. math::

        K = \\begin{bmatrix} 0 & \\cdots & 0 & 1 \\end{bmatrix}
            \\mathcal{C}^{-1} \\phi(A)

    where :math:`\\phi(s) = \\prod (s - p_i)` and :math:`\\mathcal{C}` is
    the controllability matrix.

    :param A: (n, n) state matrix.
    :param B: (n, 1) or (n,) input matrix.
    :param poles: Desired closed-loop pole locations (length n).
    :return: DescriptiveResult with gain vector K and closed-loop eigenvalues.
    :raises ValueError: If system is not controllable or dimensions mismatch.

    References
    ----------
    Ackermann, J. (1972). Der Entwurf linearer Regelungssysteme im
    Zustandsraum. *Regelungstechnik*, 20, 297-300.
    Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    poles = np.asarray(poles, dtype=np.complex128)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = A.shape[0]
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.shape[0] != n:
        raise ValueError(f"B rows ({B.shape[0]}) must match A rows ({n}).")
    if len(poles) != n:
        raise ValueError(f"Need {n} poles, got {len(poles)}.")

    C_mat = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    if np.linalg.matrix_rank(C_mat) < n:
        raise ValueError("System is not controllable.")

    char_poly = np.real(np.poly(poles))
    phi_A = np.zeros((n, n), dtype=np.float64)
    for i, c in enumerate(char_poly):
        phi_A += c * np.linalg.matrix_power(A, n - i)

    e_n = np.zeros(n)
    e_n[-1] = 1.0
    K = (e_n @ np.linalg.inv(C_mat) @ phi_A).flatten()

    cl_eigs = np.linalg.eigvals(A - B @ K.reshape(1, -1))

    return DescriptiveResult(
        name="state_feedback",
        value=float(np.max(np.abs(K))),
        extra={
            "K": K,
            "closed_loop_eigenvalues": cl_eigs,
            "desired_poles": poles,
            "controllability_rank": int(np.linalg.matrix_rank(C_mat)),
        },
    )


ssfbk = state_feedback


def cheatsheet() -> str:
    return "state_feedback({}) -> State feedback gain via pole placement."
