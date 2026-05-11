# morie.fn — function file (hadesllm/morie)
"""R-hat (Gelman-Rubin) convergence diagnostic."""

from __future__ import annotations

__all__ = ["rhat_diagnostic", "rhatd"]

from typing import Any, Union

import numpy as np


def rhat_diagnostic(
    chains: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Gelman-Rubin R-hat convergence diagnostic for multiple chains.

    Compares within-chain and between-chain variance.  R-hat near 1
    indicates convergence; values > 1.01 suggest the chains have not
    mixed.  If a single chain is provided, it is split in half.

    .. math::

        \\hat{R} = \\sqrt{\\frac{\\hat{V}}{W}}

    where W is the within-chain variance and hat{V} is the marginal
    posterior variance estimate incorporating between-chain variance B.

    Parameters
    ----------
    chains : array-like
        Shape (m, n) for m chains of length n, or (n,) for a single
        chain that will be split.

    Returns
    -------
    dict
        rhat, converged (rhat < 1.01), W, B, n_chains, n_samples

    References
    ----------
    Gelman, A. & Rubin, D. B. (1992). Inference from iterative
    simulation using multiple sequences. *Statistical Science*,
    7(4), 457--472.
    Vehtari, A., et al. (2021). Rank-normalization, folding, and
    localization. *Bayesian Analysis*, 16(2), 667--718.
    """
    if isinstance(chains, list) and len(chains) > 0 and isinstance(chains[0], (list, np.ndarray)):
        chains_arr = np.array([np.asarray(c, dtype=float) for c in chains])
    else:
        chains_arr = np.asarray(chains, dtype=float)

    if chains_arr.ndim == 1:
        mid = len(chains_arr) // 2
        chains_arr = np.array([chains_arr[:mid], chains_arr[mid: 2 * mid]])

    m, n = chains_arr.shape
    if m < 2:
        raise ValueError("Need at least 2 chains (or a single chain long enough to split).")

    chain_means = np.mean(chains_arr, axis=1)
    grand_mean = np.mean(chain_means)

    B = n * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    W = float(np.mean(np.var(chains_arr, axis=1, ddof=1)))

    var_hat = (1.0 - 1.0 / n) * W + B / n
    rhat = float(np.sqrt(var_hat / W)) if W > 1e-30 else 1.0

    return {
        "rhat": rhat,
        "converged": bool(rhat < 1.01),
        "W": float(W),
        "B": float(B),
        "n_chains": int(m),
        "n_samples": int(n),
    }


rhatd = rhat_diagnostic


def cheatsheet() -> str:
    return "rhat_diagnostic(chains) -> R-hat (Gelman-Rubin) convergence diagnostic."
