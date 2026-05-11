# morie.fn — function file (hadesllm/morie)
"""Gradient statistics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gradient_stats(
    grads: list[np.ndarray],
) -> DescriptiveResult:
    """Compute summary statistics of gradient tensors.

    Reports global norm, max, min, mean, and sparsity (fraction of zeros).

    :param grads: List of gradient arrays.
    :return: DescriptiveResult with gradient statistics.
    """
    if not grads:
        raise ValueError("grads must not be empty")

    all_vals = np.concatenate([g.ravel() for g in grads])
    global_norm = float(np.sqrt(np.sum(all_vals**2)))
    per_layer_norms = [float(np.sqrt(np.sum(g**2))) for g in grads]

    return DescriptiveResult(
        name="gradient_stats",
        value=global_norm,
        extra={
            "global_norm": global_norm,
            "max": float(np.max(all_vals)),
            "min": float(np.min(all_vals)),
            "mean": float(np.mean(all_vals)),
            "std": float(np.std(all_vals)),
            "sparsity": float(np.mean(np.abs(all_vals) < 1e-10)),
            "per_layer_norms": per_layer_norms,
            "n_params": len(all_vals),
        },
    )


def cheatsheet() -> str:
    return "gradient_stats(grads) -> norm, max, min, sparsity of gradients"


grdst = gradient_stats
