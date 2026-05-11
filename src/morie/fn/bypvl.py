# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian p-value."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_p_value(chain, test_stat) -> DescriptiveResult:
    """Bayesian p-value: proportion of posterior exceeding test_stat.

    .. epigraph:: "What is dead may never die." -- Ironborn, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float).ravel()
    bp = float(np.mean(chain >= test_stat))
    return DescriptiveResult(
        name="bayesian_p_value",
        value=bp,
        extra={
            "bayesian_p": bp,
            "test_stat": float(test_stat),
            "n_samples": len(chain),
            "chain_mean": float(np.mean(chain)),
        },
    )


bypvl = bayesian_p_value


def cheatsheet() -> str:
    return "bayesian_p_value({}) -> Bayesian p-value."
