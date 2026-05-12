# morie.fn -- function file (hadesllm/morie)
"""Model-based clustering (BIC selection)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from .gmmcl import gmm_cluster


def model_based_cluster(
    data: np.ndarray,
    k_range: tuple[int, int] = (2, 10),
    seed: int = 42,
) -> DescriptiveResult:
    """Model-based clustering: select optimal k via BIC over GMM fits.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    k_range : tuple
        (min_k, max_k) range.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is labels from the best model.
        ``extra`` has ``optimal_k``, ``bic_values``, ``aic_values``.
    """
    X = np.asarray(data, dtype=np.float64)
    k_min, k_max = k_range
    ks = list(range(k_min, k_max + 1))

    bics = []
    aics = []
    results = []

    for k in ks:
        res = gmm_cluster(X, n_components=k, seed=seed)
        bics.append(res.extra["bic"])
        aics.append(res.extra["aic"])
        results.append(res)

    best_idx = int(np.argmin(bics))
    best = results[best_idx]

    return DescriptiveResult(
        name="ModelBasedClustering",
        value=best.value,
        extra={
            "optimal_k": ks[best_idx],
            "bic_values": dict(zip(ks, bics)),
            "aic_values": dict(zip(ks, aics)),
            "means": best.extra["means"],
        },
    )


mclst = model_based_cluster


def cheatsheet() -> str:
    return "model_based_cluster({}) -> Model-based clustering with BIC selection."
