# morie.fn -- function file (hadesllm/morie)
r"""Rotary positional encoding (RoPE).

Applies rotation matrices to position-encode via complex numbers.

References
----------
Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021).
RoFormer: Enhanced transformer with rotary position embedding.
arXiv preprint arXiv:2104.09864.
"""

__all__ = ["rpenc"]

import numpy as np


def rpenc(seq_len, d_model, base=10000):
    """
    Rotary positional encoding.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    d_model : int
        Model dimension (must be even).
    base : float, optional
        Base for frequency. Default 10000.

    Returns
    -------
    ndarray
        RoPE encodings, shape (seq_len, d_model).
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even")

    positions = np.arange(seq_len)
    frequencies = 1.0 / np.power(base, np.arange(0, d_model, 2) / d_model)

    angles = np.outer(positions, frequencies)

    rope = np.zeros((seq_len, d_model))
    rope[:, 0::2] = np.cos(angles)
    rope[:, 1::2] = np.sin(angles)

    return rope
