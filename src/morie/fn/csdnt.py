# morie.fn — function file (hadesllm/morie)
"""Compressed sensing denoising via L1 regularization."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope. -- Jyn Erso"


def denoise_cs(
    y,
    A,
    lambda_: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    **kwargs,
) -> DescriptiveResult:
    r"""
    Denoise a signal using compressed sensing L1 regularization (ISTA).

    Solves the basis pursuit denoising (BPDN) problem:

    .. math::

        \\min_x \\frac{1}{2}\\|y - Ax\\|_2^2 + \\lambda \\|x\\|_1

    Applicable when the signal has a sparse representation under *A*
    (dictionary/basis). The denoised signal is :math:`\\hat{y} = A\\hat{x}`.

    :param y: (m,) noisy observation vector.
    :param A: (m, n) dictionary/sensing matrix.
    :param lambda_: Regularization parameter. Default 0.1.
    :param max_iter: Maximum iterations. Default 500.
    :param tol: Convergence tolerance. Default 1e-6.
    :return: DescriptiveResult with denoised signal and SNR improvement.

    References
    ----------
    Chen, S. S., Donoho, D. L. & Saunders, M. A. (2001). Atomic
    decomposition by basis pursuit. *SIAM Review*, 43(1), 129-159.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape

    if len(y) != m:
        raise ValueError(f"y length ({len(y)}) must match A rows ({m}).")

    AtA = A.T @ A
    Aty = A.T @ y
    L = np.linalg.norm(AtA, 2)
    t = 1.0 / L if L > 0 else 1.0

    x = np.zeros(n)
    for i in range(max_iter):
        grad = AtA @ x - Aty
        x_new = np.sign(x - t * grad) * np.maximum(np.abs(x - t * grad) - t * lambda_, 0.0)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    y_denoised = A @ x
    noise_est = y - y_denoised
    snr_input = 10 * np.log10(np.sum(y**2) / (np.sum(noise_est**2) + 1e-300))

    return DescriptiveResult(
        name="denoise_cs",
        value=float(snr_input),
        extra={
            "denoised": y_denoised,
            "coefficients": x,
            "snr_db": float(snr_input),
            "residual_norm": float(np.linalg.norm(noise_est)),
            "lambda": lambda_,
            "nnz": int(np.sum(np.abs(x) > 1e-10)),
            "iterations": min(i + 1, max_iter),
        },
    )


csdnt = denoise_cs


def cheatsheet() -> str:
    return "denoise_cs({}) -> Compressed sensing denoising via L1 regularization."
