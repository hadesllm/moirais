"""SwiGLU activation."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from ._containers import DescriptiveResult


def swiglu(
    x: np.ndarray,
    W1: np.ndarray | None = None,
    W2: np.ndarray | None = None,
    W3: np.ndarray | None = None,
) -> DescriptiveResult:
    r"""Compute the SwiGLU activation (Shazeer 2020).

    :math:`\\text{SwiGLU}(x) = (x W_1 \\odot \\text{SiLU}(x W_3)) W_2`

    If weight matrices are None, operates element-wise as SiLU gating.

    :param x: Input array.
    :param W1: Gate projection weights.
    :param W2: Output projection weights.
    :param W3: Up projection weights.
    :return: DescriptiveResult with output in ``extra['output']``.
    """
    if W1 is not None and W3 is not None:
        gate = x @ W3
        up = x @ W1
    else:
        gate = x
        up = x

    silu_gate = gate * expit(gate)
    hidden = up * silu_gate

    if W2 is not None:
        output = hidden @ W2
    else:
        output = hidden

    return DescriptiveResult(
        name="swiglu",
        value=float(np.mean(output)),
        extra={"output": output, "hidden_mean": float(np.mean(hidden))},
    )


def cheatsheet() -> str:
    return "swiglu(x, W1, W2, W3) -> SwiGLU activation output"


swglu = swiglu
