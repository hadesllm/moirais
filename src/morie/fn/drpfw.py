# morie.fn — function file (hadesllm/morie)
"""Dropout forward pass (inverted dropout)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["dropout_forward"]


def dropout_forward(x, p: float = 0.5, seed: int = 0, training: bool = True):
    """Dropout forward pass with inverted scaling.

    During training, mask :math:`m \\sim \\text{Bernoulli}(1-p)` and
    :math:`y = x \\odot m / (1-p)` so :math:`\\mathbb{E}[y] = x`.  At
    inference (``training=False``) the input is passed through unchanged.

    Parameters
    ----------
    x : array-like
        Input.
    p : float
        Drop probability, in ``[0, 1)``.
    seed : int
        RNG seed.
    training : bool
        Apply dropout if True; pass-through otherwise.

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``mask``, ``p``, ``kept_fraction``.

    References
    ----------
    Srivastava, N. et al. (2014). Dropout. *JMLR* 15:1929-1958.
    """
    x = np.asarray(x, dtype=float)
    if not (0.0 <= p < 1.0):
        raise ValueError(f"p must be in [0, 1), got {p}.")
    if not training or p == 0.0:
        return RichResult(
            title="Dropout (inference)",
            summary_lines=[("p", p), ("training", training)],
            payload={"y": x, "estimate": x, "mask": np.ones_like(x), "p": p,
                     "kept_fraction": 1.0,
                     "method": "Dropout (pass-through)"},
        )
    rng = np.random.default_rng(seed)
    mask = (rng.random(x.shape) >= p).astype(x.dtype)
    y = x * mask / (1.0 - p)
    return RichResult(
        title="Dropout (forward)",
        summary_lines=[("p", p), ("kept fraction", float(mask.mean()))],
        payload={
            "y": y,
            "estimate": y,
            "mask": mask,
            "p": p,
            "kept_fraction": float(mask.mean()),
            "method": "Dropout forward (inverted)",
        },
    )


# CANONICAL TEST
# dropout_forward(np.ones(10000), p=0.5, seed=0).kept_fraction ~ 0.5
# mean(y) ~ 1.0 (E[y]=x property)


def cheatsheet():
    return "drpfw: Inverted dropout y = x*mask/(1-p), mask~Bern(1-p)"
