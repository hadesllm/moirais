# moirais.fn — function file (hadesllm/moirais)
"""Learning rate scheduler."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lr_schedule(
    step: int,
    warmup: int = 100,
    total: int = 10000,
    type: str = "cosine",
    base_lr: float = 3e-4,
    min_lr: float = 1e-5,
) -> DescriptiveResult:
    """Compute learning rate at a given training step.

    Supports cosine, linear, and constant schedules with optional warmup.

    :param step: Current training step.
    :param warmup: Number of warmup steps.
    :param total: Total number of training steps.
    :param type: Schedule type: 'cosine', 'linear', or 'constant'.
    :param base_lr: Peak learning rate after warmup.
    :param min_lr: Minimum learning rate (for cosine/linear decay).
    :return: DescriptiveResult with the learning rate.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    if step < warmup and warmup > 0:
        lr = base_lr * step / warmup
    elif type == "cosine":
        decay_steps = max(total - warmup, 1)
        progress = min((step - warmup) / decay_steps, 1.0)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
    elif type == "linear":
        decay_steps = max(total - warmup, 1)
        progress = min((step - warmup) / decay_steps, 1.0)
        lr = base_lr + (min_lr - base_lr) * progress
    else:
        lr = base_lr

    return DescriptiveResult(
        name="lr_schedule",
        value=float(lr),
        extra={"step": step, "warmup": warmup, "total": total, "type": type, "base_lr": base_lr, "min_lr": min_lr},
    )


def cheatsheet() -> str:
    return "lr_schedule(step, warmup, total, type) -> learning rate at step"


lrsch = lr_schedule
