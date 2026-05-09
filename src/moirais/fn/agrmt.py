# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Agreement score computation for legislative/spatial data."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def agreement_score(vote_matrix) -> DescriptiveResult:
    """Compute pairwise agreement scores between legislators.

    Agreement = proportion of shared votes where both legislators
    voted the same way. Returns the full agreement matrix plus
    summary statistics.

    :param vote_matrix: (n_legislators x n_votes) binary matrix.
    :return: DescriptiveResult with agreement matrix and statistics.

    References
    ----------
    Armstrong (2014), Ch 7.

    .. epigraph:: "There is always a bigger fish." -- Qui-Gon, Star Wars
    """
    V = np.asarray(vote_matrix, dtype=float)
    if V.ndim != 2:
        raise ValueError("vote_matrix must be 2D.")
    n_leg = V.shape[0]

    agree = np.eye(n_leg)
    n_shared = np.zeros((n_leg, n_leg), dtype=int)
    for i in range(n_leg):
        for j in range(i + 1, n_leg):
            mask = ~np.isnan(V[i]) & ~np.isnan(V[j])
            n_sh = int(mask.sum())
            n_shared[i, j] = n_shared[j, i] = n_sh
            if n_sh > 0:
                a = float(np.mean(V[i, mask] == V[j, mask]))
                agree[i, j] = agree[j, i] = a

    upper = agree[np.triu_indices(n_leg, k=1)]
    return DescriptiveResult(
        name="agreement_score",
        value={"agreement_matrix": agree},
        extra={
            "n_legislators": n_leg,
            "mean_agreement": float(upper.mean()) if len(upper) > 0 else 0.0,
            "min_agreement": float(upper.min()) if len(upper) > 0 else 0.0,
            "max_agreement": float(upper.max()) if len(upper) > 0 else 0.0,
            "n_shared_votes": n_shared,
        },
    )


agrmt = agreement_score


def cheatsheet() -> str:
    return "agreement_score({}) -> Pairwise agreement score computation."
