# morie.fn — function file (hadesllm/morie)
"""Gradient clipping by global norm (Pascanu et al. 2013)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["gradient_clipping"]


def gradient_clipping(x, max_norm: float = 1.0):
    """Global-norm gradient clipping.

    Formula:
        g <- g * min(1, max_norm / ||g||)
    i.e. if ``||g|| > max_norm``, scale ``g`` so its norm equals
    ``max_norm``; otherwise leave unchanged.

    Parameters
    ----------
    x : array-like or sequence of arrays
        Gradient tensor(s).  A flat array OR a list of arrays (treated
        jointly as one global vector — standard PyTorch behaviour).
    max_norm : float

    Returns
    -------
    RichResult with keys: tensor (clipped grad, list if input was list),
    clip_coef, total_norm.
    """
    if isinstance(x, (list, tuple)):
        flats = [np.asarray(g, dtype=float).ravel() for g in x]
        cat = np.concatenate(flats) if flats else np.array([], dtype=float)
    else:
        cat = np.asarray(x, dtype=float).ravel()
    total_norm = float(np.linalg.norm(cat))
    coef = float(min(1.0, max_norm / (total_norm + 1e-12)))
    if isinstance(x, (list, tuple)):
        clipped = [np.asarray(g, dtype=float) * coef for g in x]
    else:
        clipped = np.asarray(x, dtype=float) * coef
    return RichResult(
        title="Gradient Clipping by Global Norm (Pascanu 2013)",
        summary_lines=[("||g||", total_norm),
                       ("max_norm", max_norm),
                       ("clip_coef", coef)],
        payload={"tensor": clipped, "clip_coef": coef,
                 "total_norm": total_norm, "max_norm": max_norm,
                 "method": "global-norm-clip"},
    )


def cheatsheet():
    return "grdcl(grad, max_norm): global-norm gradient clip"


# CANONICAL TEST
# >>> r = gradient_clipping([3.0, 4.0], max_norm=1.0)
# >>> bool(np.isclose(np.linalg.norm(r["tensor"]), 1.0))
# True
