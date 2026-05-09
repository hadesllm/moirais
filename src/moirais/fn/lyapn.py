# moirais.fn — function file (hadesllm/moirais)
"""Lyapunov stability analysis."""

import numpy as np
from scipy import linalg

from ._containers import DescriptiveResult
def lyapunov_stability(A, Q=None, **kwargs) -> DescriptiveResult:
    """
    Assess Lyapunov stability of a linear time-invariant system.

    For :math:`\\dot{x} = Ax`, the system is asymptotically stable iff all
    eigenvalues of A have negative real parts. Additionally solves the
    continuous Lyapunov equation:

    .. math::

        A^T P + P A = -Q

    for positive definite Q (default: identity). If P is positive definite,
    :math:`V(x) = x^T P x` is a valid Lyapunov function.

    :param A: (n, n) state matrix.
    :param Q: (n, n) positive definite matrix. Default: identity.
    :return: DescriptiveResult with stability verdict and Lyapunov matrix P.
    :raises ValueError: If A is not square.

    References
    ----------
    Lyapunov, A. M. (1892). The General Problem of the Stability of Motion.
    Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.). Prentice Hall.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = A.shape[0]

    if Q is None:
        Q = np.eye(n)
    else:
        Q = np.asarray(Q, dtype=np.float64)

    eigenvalues = np.linalg.eigvals(A)
    max_real = float(np.max(np.real(eigenvalues)))
    is_stable = bool(max_real < 0)
    is_marginal = bool(np.isclose(max_real, 0, atol=1e-12))

    P = None
    p_positive_definite = False
    if is_stable:
        P = linalg.solve_continuous_lyapunov(A.T, -Q)
        p_eigs = np.linalg.eigvalsh(P)
        p_positive_definite = bool(np.all(p_eigs > 0))

    verdict = "asymptotically_stable" if is_stable else ("marginally_stable" if is_marginal else "unstable")

    return DescriptiveResult(
        name="lyapunov_stability",
        value=max_real,
        extra={
            "verdict": verdict,
            "is_stable": is_stable,
            "is_marginal": is_marginal,
            "max_real_eigenvalue": max_real,
            "eigenvalues": eigenvalues,
            "P": P,
            "P_positive_definite": p_positive_definite,
        },
    )


lyapn = lyapunov_stability


def cheatsheet() -> str:
    return "lyapunov_stability({}) -> Lyapunov stability analysis."
