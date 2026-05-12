# morie.fn -- function file (hadesllm/morie)
"""Early stopping check."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def early_stopping(
    val_losses: list[float] | np.ndarray,
    patience: int = 5,
    min_delta: float = 0.0,
) -> DescriptiveResult:
    """Check whether training should stop based on validation loss.

    Triggers when validation loss has not improved by at least *min_delta*
    for *patience* consecutive epochs.

    :param val_losses: Sequence of validation losses (one per epoch).
    :param patience: Number of epochs to wait for improvement.
    :param min_delta: Minimum change to count as improvement.
    :return: DescriptiveResult with ``value=True`` if should stop.
    """
    val_losses = np.asarray(val_losses, dtype=np.float64)
    if len(val_losses) == 0:
        raise ValueError("val_losses must not be empty")

    best_loss = val_losses[0]
    best_epoch = 0
    for i in range(1, len(val_losses)):
        if val_losses[i] < best_loss - min_delta:
            best_loss = val_losses[i]
            best_epoch = i

    epochs_without_improvement = len(val_losses) - 1 - best_epoch
    should_stop = epochs_without_improvement >= patience

    return DescriptiveResult(
        name="early_stopping",
        value=should_stop,
        extra={
            "best_loss": float(best_loss),
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "patience": patience,
        },
    )


def cheatsheet() -> str:
    return "early_stopping(val_losses, patience) -> True if should stop"


earlm = early_stopping
