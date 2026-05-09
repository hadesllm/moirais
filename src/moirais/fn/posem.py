# moirais.fn — function file (hadesllm/moirais)
r"""Positional encoding (sinusoidal).

Encodes position information into fixed vectors.

References
----------
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
Attention is all you need.
In NIPS (pp. 5998-6008).
"""

__all__ = ["posem"]

import numpy as np


def posem(seq_len, d_model, base=10000):
    """
    Sinusoidal positional encoding.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    d_model : int
        Model dimension.
    base : float, optional
        Base for positional encoding. Default 10000.

    Returns
    -------
    ndarray
        Positional encodings, shape (seq_len, d_model).
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1.0 / np.power(base, (2 * (dimensions // 2)) / np.float32(d_model))

    pos_encoding = positions * angle_rates

    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

    return pos_encoding
