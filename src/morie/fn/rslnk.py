# morie.fn — function file (hadesllm/morie)
"""Residual / skip connection."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ._richresult import RichResult

__all__ = ["residual_connection"]


def residual_connection(x, f: "Optional[Callable[[np.ndarray], np.ndarray]]" = None):
    """Residual (identity-shortcut) connection.

    .. math::

        y = \\mathcal{F}(x) + x

    where :math:`\\mathcal{F}` is an arbitrary callable layer.  If
    ``f`` is ``None``, :math:`\\mathcal{F}` defaults to the identity
    (so :math:`y = 2x`).

    Parameters
    ----------
    x : array-like
        Input.
    f : callable, optional
        The residual branch. Must return an array shape-compatible with
        ``x``.

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``Fx``.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual
    learning for image recognition. *CVPR*.
    """
    x = np.asarray(x, dtype=float)
    Fx = x if f is None else np.asarray(f(x), dtype=float)
    if Fx.shape != x.shape:
        raise ValueError(
            f"Residual branch shape {Fx.shape} != identity shape {x.shape}."
        )
    y = Fx + x
    return RichResult(
        title="Residual connection",
        summary_lines=[("shape", x.shape)],
        payload={
            "y": y,
            "estimate": y,
            "Fx": Fx,
            "method": "Residual identity shortcut",
        },
    )


# CANONICAL TEST
# residual_connection([1,2,3], f=lambda x: x*2).y -> [3,6,9]


def cheatsheet():
    return "rslnk: Residual y = F(x) + x"
