# morie.fn — function file (hadesllm/morie)
"""Rotary position embedding / RoPE (Su et al. 2021)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rotary_position_embedding"]


def rotary_position_embedding(x, base: float = 10000.0):
    r"""Rotary positional embedding (RoPE).

    Each adjacent pair of features (2i, 2i+1) at position ``pos`` is
    rotated by angle :math:`\\theta_{pos,i} = pos / N^{2i/d}`:

    .. math::

        \\begin{pmatrix} x'_{2i} \\\\ x'_{2i+1} \\end{pmatrix}
        =
        \\begin{pmatrix} \\cos\\theta & -\\sin\\theta \\\\
                         \\sin\\theta &  \\cos\\theta \\end{pmatrix}
        \\begin{pmatrix} x_{2i} \\\\ x_{2i+1} \\end{pmatrix}

    Parameters
    ----------
    x : array-like, shape ``(seq_len, d_model)``
        Input embeddings. ``d_model`` must be even.
    base : float
        Frequency base. Default 10000.

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``angles``.

    References
    ----------
    Su, J. et al. (2021). RoFormer: enhanced transformer with rotary
    position embedding. *arXiv:2104.09864*.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    seq_len, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"d_model must be even for RoPE, got {d}.")
    half = d // 2

    pos = np.arange(seq_len)[:, None].astype(float)
    idx = np.arange(half)[None, :].astype(float)
    inv_freq = 1.0 / np.power(base, (2.0 * idx) / d)
    angles = pos * inv_freq

    cos = np.cos(angles)
    sin = np.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    y = np.empty_like(x)
    y[:, 0::2] = y_even
    y[:, 1::2] = y_odd

    return RichResult(
        title=f"RoPE ({seq_len}x{d})",
        summary_lines=[("seq_len", seq_len), ("d_model", d), ("base", base)],
        payload={
            "y": y,
            "estimate": y,
            "angles": angles,
            "method": "Rotary position embedding",
        },
    )


# CANONICAL TEST
# rotary_position_embedding([[1,0,1,0]]).y[0] == [1,0,1,0] (pos=0 -> identity)


def cheatsheet():
    return "rotrp: RoPE rotates feature pairs by theta=pos/N^(2i/d)"
