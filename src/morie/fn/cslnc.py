# morie.fn — function file (hadesllm/morie)
"""Cosine LR schedule with warmup (Loshchilov & Hutter 2017)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["cosine_lr_schedule"]


def cosine_lr_schedule(x, lr_max: float = 1e-3, lr_min: float = 0.0,
                      total_steps: int = 1000, warmup_steps: int = 0):
    """Cosine learning-rate schedule with optional linear warmup.

    Formula:
        warmup (t < warmup):   lr = lr_max * t / warmup
        cosine (t >= warmup):  lr = lr_min + 0.5 * (lr_max - lr_min) *
                                   (1 + cos(pi * (t - warmup) /
                                             (total - warmup)))

    Parameters
    ----------
    x : int or array-like of int
        Step index/indices.
    lr_max, lr_min : float
        Peak and trough learning rates.
    total_steps : int
        Total training steps.
    warmup_steps : int
        Linear-warmup steps.  0 disables warmup.

    Returns
    -------
    RichResult with keys: value (scalar lr if x scalar) OR tensor (array
    of lr) plus step.
    """
    t = np.asarray(x, dtype=float)
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must exceed warmup_steps")
    scalar = (t.ndim == 0)
    t = np.atleast_1d(t)
    lr = np.empty_like(t)
    warm = t < warmup_steps
    lr[warm] = lr_max * t[warm] / max(1.0, float(warmup_steps))
    dec = (t - warmup_steps) / (total_steps - warmup_steps)
    dec = np.clip(dec, 0.0, 1.0)
    lr[~warm] = lr_min + 0.5 * (lr_max - lr_min) * (
        1.0 + np.cos(np.pi * dec[~warm])
    )
    if scalar:
        lr_out = float(lr[0])
        tensor = np.asarray(lr_out)
    else:
        lr_out = float(lr[0])
        tensor = lr
    return RichResult(
        title="Cosine LR Schedule (Loshchilov 2017)",
        summary_lines=[("lr_max", lr_max), ("lr_min", lr_min),
                       ("total", total_steps), ("warmup", warmup_steps)],
        payload={"value": lr_out, "tensor": tensor,
                 "step": t, "lr_max": lr_max, "lr_min": lr_min,
                 "total_steps": total_steps, "warmup_steps": warmup_steps,
                 "method": "cosine-LR"},
    )


def cheatsheet():
    return "cslnc(step, lr_max, lr_min, total, warmup): cosine LR schedule"


# CANONICAL TEST
# >>> r = cosine_lr_schedule(0, lr_max=1.0, lr_min=0.0,
# ...                        total_steps=10, warmup_steps=0)
# >>> bool(np.isclose(float(r["value"]), 1.0))
# True
