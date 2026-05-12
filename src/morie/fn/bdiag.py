# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""MCMC diagnostics (Rhat, ESS)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mcmc_diagnostics(
    chains: np.ndarray | list,
) -> DescriptiveResult:
    """
    Compute MCMC convergence diagnostics: Rhat and effective sample size.

    Parameters
    ----------
    chains : array-like
        Shape (n_chains, n_iter).

    Returns
    -------
    DescriptiveResult
        extra has 'rhat', 'ess', 'n_chains', 'n_iter'.

    References
    ----------
    Gelman, A., & Rubin, D. B. (1992). Inference from iterative
    simulation using multiple sequences. *Stat Sci*, 7(4), 457-472.
    """
    c = np.asarray(chains, dtype=float)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if c.ndim != 2:
        raise ValueError("chains must be 2-D (n_chains x n_iter).")

    m, n = c.shape

    chain_means = np.mean(c, axis=1)
    grand_mean = np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1) if m > 1 else 0.0
    W = np.mean(np.var(c, axis=1, ddof=1))

    if m > 1 and W > 0:
        var_hat = (n - 1) / n * W + B / n
        rhat = np.sqrt(var_hat / W)
    else:
        rhat = 1.0

    def _ess_single(x):
        n_ = len(x)
        x_c = x - np.mean(x)
        acf = np.correlate(x_c, x_c, mode="full")[n_ - 1 :]
        acf = acf / acf[0] if acf[0] > 0 else acf
        tau = 1.0
        for k in range(1, n_ // 2):
            if acf[k] < 0:
                break
            tau += 2 * acf[k]
        return n_ / tau

    ess_per_chain = [_ess_single(c[i]) for i in range(m)]
    ess = float(np.sum(ess_per_chain))

    return DescriptiveResult(
        name="mcmc_diagnostics",
        value=float(rhat),
        extra={
            "rhat": float(rhat),
            "ess": float(ess),
            "n_chains": m,
            "n_iter": n,
        },
    )


bdiag = mcmc_diagnostics


def cheatsheet() -> str:
    return "mcmc_diagnostics({}) -> MCMC diagnostics (Rhat, ESS)."
