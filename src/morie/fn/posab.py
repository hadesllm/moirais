# morie.fn -- function file (hadesllm/morie)
"""Absolute sinusoidal positional encoding (Vaswani et al. 2017)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["positional_encoding_abs"]


def positional_encoding_abs(seq_len: int, d_model: int, base: float = 10000.0):
    r"""Absolute sinusoidal positional encoding.

    .. math::

        \\text{PE}(pos, 2i)   &= \\sin(pos / N^{2i/d}) \\\\
        \\text{PE}(pos, 2i+1) &= \\cos(pos / N^{2i/d})

    with :math:`N = 10000` by default.

    Parameters
    ----------
    seq_len : int
        Sequence length (number of positions).
    d_model : int
        Embedding dimension.
    base : float
        Frequency base ``N``. Default 10000.

    Returns
    -------
    result : RichResult
        Keys: ``PE`` / ``estimate`` of shape ``(seq_len, d_model)``.

    References
    ----------
    Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
    """
    seq_len = int(seq_len)
    d_model = int(d_model)
    if seq_len <= 0 or d_model <= 0:
        raise ValueError("seq_len and d_model must be > 0.")
    pos = np.arange(seq_len)[:, None].astype(float)          # (seq, 1)
    i = np.arange(d_model)[None, :].astype(float)             # (1, d)
    # 2i/d using floor-division of column index by 2
    div_term = np.power(base, (2.0 * (i // 2)) / d_model)     # (1, d)
    angles = pos / div_term                                   # (seq, d)
    PE = np.empty_like(angles)
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return RichResult(
        title=f"Sinusoidal positional encoding ({seq_len}x{d_model})",
        summary_lines=[("seq_len", seq_len), ("d_model", d_model),
                       ("base", base)],
        payload={
            "PE": PE,
            "estimate": PE,
            "seq_len": seq_len,
            "d_model": d_model,
            "method": "Sinusoidal positional encoding",
        },
    )


# CANONICAL TEST
# positional_encoding_abs(1, 4).PE[0] -> [sin(0), cos(0), sin(0), cos(0)]
#                                     = [0, 1, 0, 1]


def cheatsheet():
    return "posab: PE(pos,2i)=sin(pos/N^(2i/d)), PE(pos,2i+1)=cos(...)"
