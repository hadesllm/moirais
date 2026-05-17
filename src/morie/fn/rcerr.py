# morie.fn -- function file (hadesllm/morie)
"""Roll-call classification errors."""

from __future__ import annotations

from ._containers import DescriptiveResult


def roll_call_errors(predicted, observed) -> DescriptiveResult:
    """Count total, type I, and type II errors.

    .. epigraph:: Measure what is measurable, and make measurable what is not. -- Galileo Galilei
    """
    import numpy as np

    pred = np.asarray(predicted, dtype=float).round()
    obs = np.asarray(observed, dtype=float)
    total_err = int(np.sum(pred != obs))
    type_i = int(np.sum((pred == 1) & (obs == 0)))
    type_ii = int(np.sum((pred == 0) & (obs == 1)))
    n = len(obs)
    return DescriptiveResult(
        name="roll_call_errors",
        value=float(total_err),
        extra={
            "total_errors": total_err,
            "type_i": type_i,
            "type_ii": type_ii,
            "n": n,
            "error_rate": float(total_err / max(n, 1)),
        },
    )


rcerr = roll_call_errors


def cheatsheet() -> str:
    return "roll_call_errors({}) -> Roll-call classification errors."
