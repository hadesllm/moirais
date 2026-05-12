# morie.fn — function file (hadesllm/morie)
"""Kolmogorov complexity approximation via compression ratio."""

__all__ = ["kolmc"]

import zlib

import numpy as np


def kolmc(data: np.ndarray) -> dict:
    r"""
    Approximate Kolmogorov complexity using compression ratio.

    K(x) is uncomputable, but the length of a compressed representation
    provides an upper bound. Uses zlib (DEFLATE) as the universal compressor.

    .. math::

        \\hat{K}(x) \\approx |C(x)| \\quad \\text{where } C
        \\text{ is a compressor}

    Parameters
    ----------
    data : np.ndarray
        Input data (integer or float array). Converted to bytes for
        compression.

    Returns
    -------
    dict
        'compressed_size' (int, bytes),
        'original_size' (int, bytes),
        'compression_ratio' (float, original / compressed),
        'normalized_complexity' (float, compressed / original, in [0, 1+]),
        'complexity_bits' (float, compressed_size * 8).

    References
    ----------
    Kolmogorov, A. N. (1965). Three approaches to the quantitative
    definition of information. Problems Inform. Transmission, 1(1), 1-7.
    Li, M. & Vitanyi, P. (2008). An Introduction to Kolmogorov Complexity
    and Its Applications. Springer, 3rd ed.
    """
    data = np.asarray(data)
    raw = data.tobytes()
    original_size = len(raw)

    if original_size == 0:
        return {
            "compressed_size": 0,
            "original_size": 0,
            "compression_ratio": 1.0,
            "normalized_complexity": 0.0,
            "complexity_bits": 0.0,
        }

    compressed = zlib.compress(raw, level=9)
    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else float("inf")
    norm = compressed_size / original_size

    return {
        "compressed_size": compressed_size,
        "original_size": original_size,
        "compression_ratio": ratio,
        "normalized_complexity": norm,
        "complexity_bits": compressed_size * 8.0,
    }
