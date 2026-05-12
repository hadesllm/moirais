# morie.fn — function file (hadesllm/morie)
"""Temperature scaling for logits (Hinton et al. 2015)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["temperature_scaling"]


def temperature_scaling(x, T: float = 1.0):
    """Temperature-scaled softmax.

    Formula: ``p_i = exp(z_i / T) / sum_j exp(z_j / T)``.

    ``T -> 0+`` peaks the distribution onto argmax; ``T -> inf`` makes
    it uniform.  Numerically stable via the max-subtract trick.

    Parameters
    ----------
    x : array-like of logits.
    T : float
        Temperature.  Must be > 0.

    Returns
    -------
    RichResult with keys: tensor (probabilities), entropy.
    """
    if T <= 0:
        raise ValueError("Temperature must be > 0")
    z = np.asarray(x, dtype=float) / T
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / np.sum(e, axis=-1, keepdims=True)
    H = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)
    return RichResult(
        title="Temperature-Scaled Softmax (Hinton 2015)",
        summary_lines=[("T", T),
                       ("entropy_mean", float(np.mean(H)))],
        payload={"tensor": p, "entropy": H, "T": T,
                 "method": "temperature-softmax"},
    )


def cheatsheet():
    return "tmpsc(logits, T): softmax with temperature T"


# CANONICAL TEST
# >>> r = temperature_scaling([1.0, 2.0, 3.0], T=1.0)
# >>> bool(np.isclose(r["tensor"].sum(), 1.0))
# True
