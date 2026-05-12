# morie.fn -- function file (hadesllm/morie)
"""Polar code construction (Bhattacharyya parameter method)."""

__all__ = ["polcd"]

import numpy as np


def polcd(
    n: int,
    k: int,
    *,
    design_snr_db: float = 0.0,
) -> dict:
    """
    Construct a polar code by selecting the k most reliable bit-channels.

    Uses Bhattacharyya parameter evolution for a BEC or AWGN channel.

    Parameters
    ----------
    n : int
        Code length, must be a power of 2.
    k : int
        Number of information bits, 0 < k <= n.
    design_snr_db : float
        Design SNR in dB for Bhattacharyya parameter initialization.

    Returns
    -------
    dict
        'info_bits' (np.ndarray, indices of k information bit-channels),
        'frozen_bits' (np.ndarray, indices of frozen bit-channels),
        'bhattacharyya' (np.ndarray, Bhattacharyya parameters for all n channels),
        'rate' (k / n).

    Raises
    ------
    ValueError
        If n is not a power of 2 or k out of range.

    References
    ----------
    Arikan, E. (2009). Channel polarization: A method for constructing
    capacity-achieving codes for symmetric binary-input memoryless channels.
    IEEE Trans. Inform. Theory, 55(7), 3051-3073.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}.")
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, n], got k={k}, n={n}.")

    snr_lin = 10.0 ** (design_snr_db / 10.0)
    z = np.exp(-snr_lin) * np.ones(1)

    m = int(np.log2(n))
    z_channels = z.copy()

    for _ in range(m):
        z_new = np.empty(2 * len(z_channels))
        for i, zi in enumerate(z_channels):
            z_minus = 2 * zi - zi * zi
            z_plus = zi * zi
            z_minus = min(z_minus, 1.0)
            z_plus = max(z_plus, 0.0)
            z_new[2 * i] = z_minus
            z_new[2 * i + 1] = z_plus
        z_channels = z_new

    sorted_indices = np.argsort(z_channels)
    info_bits = np.sort(sorted_indices[:k])
    frozen_bits = np.sort(sorted_indices[k:])

    return {
        "info_bits": info_bits,
        "frozen_bits": frozen_bits,
        "bhattacharyya": z_channels,
        "rate": k / n,
    }
