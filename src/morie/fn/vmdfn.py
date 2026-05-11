"""Variational mode decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In a dark place we find ourselves, and a little more knowledge lights our way."


def variational_mode_decompose(
    x, K: int = 3, alpha: float = 2000.0, tau: float = 0.0, tol: float = 1e-7, max_iter: int = 500, **kwargs
) -> DescriptiveResult:
    """Variational mode decomposition (VMD).

    Parameters
    ----------
    x : array-like
        1-D input signal.
    K : int
        Number of modes to extract (default 3).
    alpha : float
        Bandwidth constraint (default 2000).
    tau : float
        Noise-tolerance (Lagrangian update step, default 0).
    tol : float
        Convergence tolerance (default 1e-7).
    max_iter : int
        Maximum iterations (default 500).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of modes K; ``extra`` has ``modes`` (K x N array),
        ``center_frequencies`` (K,), ``iterations``.

    References
    ----------
    Dragomiretskiy, K., & Zosso, D. (2014). Variational mode
    decomposition. *IEEE TSP*, 62(3), 531-544.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    T = N
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1.0 / T

    f_hat = np.fft.fft(x)
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: N // 2] = 0

    u_hat = np.zeros((K, N), dtype=complex)
    omega = np.zeros((K, max_iter))
    for k in range(K):
        omega[k, 0] = (0.5 / K) * k

    lambda_hat = np.zeros(N, dtype=complex)

    iters = 0
    for n_iter in range(max_iter):
        u_hat_old = np.copy(u_hat)
        for k in range(K):
            residual = f_hat_plus - np.sum(u_hat, axis=0) + u_hat[k]
            numer = residual + lambda_hat / 2.0
            denom = 1.0 + alpha * (freqs - omega[k, n_iter]) ** 2
            u_hat[k] = numer / denom

            abs_sq = np.abs(u_hat[k][N // 2 :]) ** 2
            freq_range = freqs[N // 2 :]
            total = np.sum(abs_sq) + 1e-15
            omega_new = np.dot(freq_range, abs_sq) / total
            if n_iter + 1 < max_iter:
                omega[k, n_iter + 1] = omega_new

        lambda_hat = lambda_hat + tau * (f_hat_plus - np.sum(u_hat, axis=0))
        iters = n_iter + 1

        delta = np.sum(np.abs(u_hat - u_hat_old) ** 2) / (N + 1e-15)
        if delta < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=1))
    center_freqs = omega[:, min(iters, max_iter - 1)]

    return DescriptiveResult(
        name="variational_mode_decompose",
        value=K,
        extra={"modes": modes, "center_frequencies": center_freqs, "iterations": iters},
    )


vmdfn = variational_mode_decompose


def cheatsheet() -> str:
    return "variational_mode_decompose({}) -> Variational mode decomposition."
