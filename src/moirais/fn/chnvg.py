# moirais.fn — function file (hadesllm/moirais)
"""Chain convergence test."""

from __future__ import annotations

from ._containers import DescriptiveResult


def chain_convergence_test(chains, alpha=0.05) -> DescriptiveResult:
    """Test MCMC convergence using R-hat < 1.1 criterion.

    .. epigraph:: "A sword needs a sheath." -- Tywin, Game of Thrones
    """
    import numpy as np

    chains = [np.asarray(c, dtype=float).ravel() for c in chains]
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    chain_means = np.array([np.mean(c) for c in chains])
    grand_mean = np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1)
    W = np.mean([np.var(c, ddof=1) for c in chains])
    var_hat = (1 - 1 / n) * W + B / n
    rhat = float(np.sqrt(var_hat / max(W, 1e-15)))
    converged = rhat < 1.1
    return DescriptiveResult(
        name="chain_convergence_test",
        value=rhat,
        extra={
            "rhat": rhat,
            "converged": converged,
            "threshold": 1.1,
            "n_chains": m,
            "n_samples": n,
        },
    )


chnvg = chain_convergence_test


def cheatsheet() -> str:
    return "chain_convergence_test({}) -> Chain convergence test."
