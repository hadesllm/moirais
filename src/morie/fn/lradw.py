# morie.fn -- function file (hadesllm/morie)
"""Linear LR warmup (Vaswani et al. 2017, Transformer "Noam" schedule)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["lr_warmup"]


def lr_warmup(x, lr_target: float = 1e-3, warmup_steps: int = 1000):
    """Linear learning-rate warmup.

    Formula:  lr = lr_target * min(1, step / warmup_steps).

    Parameters
    ----------
    x : int or array-like of int
        Step index/indices.
    lr_target : float
        LR plateau after warmup.
    warmup_steps : int
        Number of warmup steps.

    Returns
    -------
    RichResult with keys: tensor (lr), value (scalar headline).
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0")
    t = np.asarray(x, dtype=float)
    frac = np.minimum(1.0, t / float(warmup_steps))
    lr = lr_target * frac
    val = float(lr) if lr.ndim == 0 else float(lr[0])
    return RichResult(
        title="Linear LR Warmup (Vaswani 2017)",
        summary_lines=[("lr_target", lr_target),
                       ("warmup_steps", warmup_steps)],
        payload={"tensor": lr, "value": val,
                 "lr_target": lr_target, "warmup_steps": warmup_steps,
                 "step": t, "method": "linear-warmup"},
    )


def cheatsheet():
    return "lradw(step, lr_target, warmup_steps): linear LR warmup"


# CANONICAL TEST
# >>> r = lr_warmup(500, lr_target=1.0, warmup_steps=1000)
# >>> bool(np.isclose(float(r["value"]), 0.5))
# True
