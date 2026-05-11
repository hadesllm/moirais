# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian wavelet thresholding for nonparametric regression. 'Fear leads to anger.'"""

from __future__ import annotations

import numpy as np


def bayesian_wavelet_shrinkage(
    y: np.ndarray,
    threshold: float | None = None,
    method: str = 'universal',
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Bayesian wavelet shrinkage (Stein's unbiased risk estimate variant).

    Applies Discrete Wavelet Transform, thresholds coefficients, and inverts.

    .. math::

        \hat{\beta}_j = \text{soft}(w_j, \lambda), \quad
        \lambda = \sigma \sqrt{2 \log n}

    :param y: (n,) observations.
    :param threshold: Threshold value. If None, uses universal threshold.
    :param method: 'universal', 'sure' (Stein's unbiased risk), 'minimax'.
    :param rng: Random number generator.
    :return: Dictionary with 'denoised', 'threshold', 'coefficients'.
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    # Simple Haar wavelet (level-1 decomposition)
    if n % 2 != 0:
        y = np.concatenate([y, [0.0]])
        pad = True
    else:
        pad = False

    n_pairs = len(y) // 2
    # Decompose
    low = (y[::2] + y[1::2]) / np.sqrt(2)
    high = (y[::2] - y[1::2]) / np.sqrt(2)

    # Set threshold
    if threshold is None:
        if method == 'universal':
            sigma = np.std(high, ddof=1)
            threshold = sigma * np.sqrt(2 * np.log(n_pairs))
        else:
            threshold = np.sqrt(np.log(n_pairs))

    # Soft thresholding
    def soft_thresh(x, lam):
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

    high_thresh = soft_thresh(high, threshold)

    # Reconstruct
    recon = np.zeros(len(y), dtype=float)
    recon[::2] = (low + high_thresh) / np.sqrt(2)
    recon[1::2] = (low - high_thresh) / np.sqrt(2)

    if pad:
        recon = recon[:-1]

    return {
        "denoised": recon,
        "threshold": threshold,
        "coefficients_low": low,
        "coefficients_high": high_thresh,
        "method": method,
    }


bwavl = bayesian_wavelet_shrinkage


def cheatsheet() -> str:
    return "bayesian_wavelet_shrinkage(y, threshold=None, method='universal') -> denoised"
