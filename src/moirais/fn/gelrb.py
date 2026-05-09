# moirais.fn — function file (hadesllm/moirais)
"""Gelman-Rubin R-hat statistic."""

from __future__ import annotations

from ._containers import DescriptiveResult


def gelman_rubin_rhat(chains) -> DescriptiveResult:
    """Compute R-hat convergence diagnostic from multiple chains.

    .. epigraph:: "Chaos is a ladder." -- Littlefinger, Game of Thrones
    """
    import numpy as np

    chains = [np.asarray(c, dtype=float) for c in chains]
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    chain_means = np.array([np.mean(c) for c in chains])
    grand_mean = np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1)
    W = np.mean([np.var(c, ddof=1) for c in chains])
    var_hat = (1 - 1 / n) * W + B / n
    rhat = float(np.sqrt(var_hat / max(W, 1e-15)))
    return DescriptiveResult(
        name="gelman_rubin_rhat",
        value=rhat,
        extra={
            "rhat": rhat,
            "B": float(B),
            "W": float(W),
            "n_chains": m,
            "n_samples": n,
        },
    )


gelrb = gelman_rubin_rhat


def cheatsheet() -> str:
    return "gelman_rubin_rhat({}) -> Gelman-Rubin R-hat statistic."
