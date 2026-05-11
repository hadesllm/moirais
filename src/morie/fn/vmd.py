"""Variational Mode Decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful the mind of a child is."


def variational_mode(
    x, K: int = 3, alpha: float = 2000.0, tau: float = 0.0, tol: float = 1e-7, max_iter: int = 500, **kwargs
) -> DescriptiveResult:
    """Variational Mode Decomposition of 1-D signal.

    Decomposes signal into *K* band-limited modes by solving a
    constrained variational problem in the spectral domain.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    K : int
        Number of modes to extract (default 3).
    alpha : float
        Bandwidth constraint penalty (default 2000).
    tau : float
        Noise tolerance / Lagrangian update step (default 0, exact).
    tol : float
        Convergence tolerance (default 1e-7).
    max_iter : int
        Maximum iterations (default 500).

    Returns
    -------
    DescriptiveResult
        ``value`` is K; ``extra`` has ``modes``, ``center_frequencies``.

    References
    ----------
    Dragomiretskiy, K., & Zosso, D. (2014). Variational mode
    decomposition. *IEEE Trans. Signal Process.*, 62(3), 531-544.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    T = N
    t = np.arange(T) / T
    freqs = np.fft.fftfreq(T)
    x_hat = np.fft.fft(x)

    u_hat = np.zeros((K, T), dtype=complex)
    omega = np.linspace(0, 0.5, K, endpoint=False)
    lam_hat = np.zeros(T, dtype=complex)

    for _ in range(max_iter):
        u_hat_old = u_hat.copy()
        for k in range(K):
            summation = np.sum(u_hat, axis=0) - u_hat[k]
            numerator = x_hat - summation + lam_hat / 2.0
            denominator = 1.0 + 2.0 * alpha * (freqs - omega[k]) ** 2
            u_hat[k] = numerator / denominator
            abs_sq = np.abs(u_hat[k]) ** 2
            total = np.sum(abs_sq) + 1e-15
            omega[k] = float(np.sum(freqs * abs_sq) / total)

        lam_hat = lam_hat + tau * (x_hat - np.sum(u_hat, axis=0))

        conv = 0.0
        for k in range(K):
            conv += np.sum(np.abs(u_hat[k] - u_hat_old[k]) ** 2)
        if conv / (np.sum(np.abs(x_hat) ** 2) + 1e-15) < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=1))
    return DescriptiveResult(
        name="variational_mode",
        value=K,
        extra={"modes": modes, "center_frequencies": omega},
    )


vmd = variational_mode


def cheatsheet() -> str:
    return "variational_mode({}) -> Variational Mode Decomposition."
