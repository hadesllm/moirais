# morie.fn -- function file (hadesllm/morie)
"""Complete EEMD with Adaptive Noise (CEEMDAN)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The whole is greater than the sum of its parts. -- Aristotle"


def _sift_one_imf(signal, max_iter=300):
    """Extract one IMF via basic sifting."""
    h = signal.copy()
    N = len(h)
    for _ in range(max_iter):
        maxima = []
        minima = []
        for i in range(1, N - 1):
            if h[i] > h[i - 1] and h[i] >= h[i + 1]:
                maxima.append(i)
            if h[i] < h[i - 1] and h[i] <= h[i + 1]:
                minima.append(i)
        if len(maxima) < 2 or len(minima) < 2:
            break
        t = np.arange(N)
        upper = np.interp(t, maxima, h[maxima])
        lower = np.interp(t, minima, h[minima])
        mean_env = (upper + lower) / 2.0
        h_new = h - mean_env
        if np.sum(mean_env**2) / (np.sum(h**2) + 1e-15) < 1e-6:
            h = h_new
            break
        h = h_new
    return h


def ceemdan(
    x, num_sifts: int = 100, noise_std: float = 0.2, seed: int | None = None, max_imfs: int = 10, **kwargs
) -> DescriptiveResult:
    """Complete EEMD with Adaptive Noise.

    Each stage adds adaptive noise and averages the resulting IMFs,
    producing a more complete decomposition than standard EEMD.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    num_sifts : int
        Number of noise realisations (default 100).
    noise_std : float
        Noise standard deviation relative to signal std (default 0.2).
    seed : int or None
        Random seed for reproducibility.
    max_imfs : int
        Maximum number of IMFs (default 10).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of IMFs; ``extra`` has ``imfs``, ``residual``.

    References
    ----------
    Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P.
    (2011). A complete ensemble empirical mode decomposition with
    adaptive noise. *ICASSP*, 4144-4147.
    """
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    N = len(x)
    sigma = noise_std * np.std(x)
    imfs = []
    residual = x.copy()
    for _ in range(max_imfs):
        if np.sum(residual**2) < 1e-12:
            break
        ensemble = np.zeros(N)
        for _ in range(num_sifts):
            noise = rng.normal(0, sigma, N)
            imf = _sift_one_imf(residual + noise)
            ensemble += imf
        ensemble /= num_sifts
        imfs.append(ensemble)
        residual = residual - ensemble
    imfs = np.array(imfs)
    return DescriptiveResult(
        name="ceemdan",
        value=len(imfs),
        extra={"imfs": imfs, "residual": residual, "num_sifts": num_sifts},
    )


ceemd = ceemdan


def cheatsheet() -> str:
    return "_sift_one_imf({}) -> Complete EEMD with Adaptive Noise (CEEMDAN)."
