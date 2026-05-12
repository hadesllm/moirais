# morie.fn -- function file (hadesllm/morie)
"""Proportional reduction in error statistic."""

from __future__ import annotations

from ._containers import DescriptiveResult


def pre_statistic(predicted, observed) -> DescriptiveResult:
    """PRE: 1 - errors_model / errors_null.

    .. epigraph:: "I am the one who knocks." -- Walter White, Breaking Bad
    """
    import numpy as np

    pred = np.asarray(predicted, dtype=float).round()
    obs = np.asarray(observed, dtype=float)
    n = len(obs)
    modal = float(np.round(np.mean(obs)))
    errors_null = int(np.sum(obs != modal))
    errors_model = int(np.sum(pred != obs))
    pre = 1.0 - (errors_model / max(errors_null, 1))
    return DescriptiveResult(
        name="pre_statistic",
        value=float(pre),
        extra={
            "errors_null": errors_null,
            "errors_model": errors_model,
            "n": n,
        },
    )


prest = pre_statistic


def cheatsheet() -> str:
    return "pre_statistic({}) -> Proportional reduction in error statistic."
