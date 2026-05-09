# moirais.fn — function file (hadesllm/moirais)
"""Logit soft-capping."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def logit_softcap(
    logits: np.ndarray,
    cap: float = 30.0,
) -> DescriptiveResult:
    """Apply logit soft-capping to prevent extreme values.

    :math:`\\text{softcap}(x) = \\text{cap} \\cdot \\tanh(x / \\text{cap})`

    Used in Gemma 2 and autoresearch GPT to stabilize training.

    :param logits: Input logits array.
    :param cap: Soft-cap value.
    :return: DescriptiveResult with capped logits in ``extra['output']``.
    """
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")

    logits = np.asarray(logits, dtype=np.float64)
    output = cap * np.tanh(logits / cap)

    return DescriptiveResult(
        name="logit_softcap",
        value=float(np.max(np.abs(output))),
        extra={
            "output": output,
            "cap": cap,
            "max_input": float(np.max(np.abs(logits))),
            "max_output": float(np.max(np.abs(output))),
        },
    )


def cheatsheet() -> str:
    return "logit_softcap(logits, cap) -> cap*tanh(logits/cap)"


logsc = logit_softcap
