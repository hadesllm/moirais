# morie.fn -- function file (hadesllm/morie)
"""L1 minimization via Iterative Shrinkage-Thresholding (ISTA)."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Hope is like the sun. -- Vice Admiral Holdo"


def basis_pursuit_l1(
    A,
    y,
    lambda_: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    **kwargs,
) -> DescriptiveResult:
    r"""
    Solve the LASSO / basis pursuit denoising problem via ISTA.

    .. math::

        \\min_x \\frac{1}{2}\\|Ax - y\\|_2^2 + \\lambda \\|x\\|_1

    The proximal gradient step is:

    .. math::

        x^{k+1} = \\text{soft}(x^k - t A^T(Ax^k - y),\\; t\\lambda)

    with step size :math:`t = 1 / \\|A^T A\\|_2`.

    :param A: (m, n) sensing matrix.
    :param y: (m,) observation vector.
    :param lambda_: Regularization parameter. Default 0.1.
    :param max_iter: Maximum ISTA iterations. Default 1000.
    :param tol: Convergence tolerance. Default 1e-6.
    :return: DescriptiveResult with sparse solution.

    References
    ----------
    Daubechies, I., Defrise, M. & De Mol, C. (2004). An iterative
    thresholding algorithm for linear inverse problems with a sparsity
    constraint. *Comm. Pure Appl. Math.*, 57(11), 1413-1457.
    Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. *SIAM J. Imaging Sci.*, 2(1), 183-202.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    m, n = A.shape

    L = np.linalg.norm(A.T @ A, 2)
    t = 1.0 / L if L > 0 else 1.0

    x = np.zeros(n)
    AtA = A.T @ A
    Aty = A.T @ y

    def _soft_threshold(z, thresh):
        return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)

    for i in range(max_iter):
        grad = AtA @ x - Aty
        x_new = _soft_threshold(x - t * grad, t * lambda_)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    residual = float(np.linalg.norm(A @ x - y))
    l1_norm = float(np.sum(np.abs(x)))

    return DescriptiveResult(
        name="basis_pursuit_l1",
        value=residual,
        extra={
            "x_recovered": x,
            "residual_norm": residual,
            "l1_norm": l1_norm,
            "objective": 0.5 * residual**2 + lambda_ * l1_norm,
            "lambda": lambda_,
            "iterations": min(i + 1, max_iter),
            "nnz": int(np.sum(np.abs(x) > 1e-10)),
        },
    )


csbp = basis_pursuit_l1


def cheatsheet() -> str:
    return "basis_pursuit_l1({}) -> L1 minimization via Iterative Shrinkage-Thresholding (ISTA)."
