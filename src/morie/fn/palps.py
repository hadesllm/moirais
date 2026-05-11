# morie.fn — function file (hadesllm/morie)
"""It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def parallel_analysis(data, *, n_sim: int = 100, seed: int = 42, alpha: float = 0.05) -> DescriptiveResult:
    """Horn's parallel analysis: how many factors to retain?

    Parameters
    ----------
    data : array-like or DataFrame
        Data matrix (n x p).
    n_sim : int
        Number of random simulations. Default 100.
    seed : int
        Random seed. Default 42.
    alpha : float
        Percentile threshold. Default 0.05.

    Returns
    -------
    DescriptiveResult
        With ``value`` as the number of factors to retain.
    """
    X = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    # Eigenvalues of actual correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    actual_eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    # Simulate random eigenvalues
    rng = np.random.default_rng(seed)
    sim_eigs = np.zeros((n_sim, p))
    for i in range(n_sim):
        R = rng.standard_normal((n, p))
        corr_r = np.corrcoef(R, rowvar=False)
        sim_eigs[i] = np.sort(np.linalg.eigvalsh(corr_r))[::-1]
    threshold = np.percentile(sim_eigs, 100 * (1 - alpha), axis=0)
    n_factors = int(np.sum(actual_eig > threshold))
    return DescriptiveResult(
        name="Parallel analysis",
        value=n_factors,
        extra={
            "n_factors": n_factors,
            "actual_eigenvalues": actual_eig.tolist(),
            "threshold": threshold.tolist(),
            "n_sim": n_sim,
        },
    )


palps = parallel_analysis


def cheatsheet() -> str:
    return "parallel_analysis({}) -> Parallel analysis for factor analysis. 'Everything is procee"
