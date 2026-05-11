# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox weight matrix estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bb_weight_matrix(Z, n_dims: int = 2) -> DescriptiveResult:
    """Estimate Blackbox weight matrix W (per-issue loadings on latent dims).

    Uses SVD of double-centered Z.

    :param Z: Respondent x issue data matrix.
    :param n_dims: Number of latent dimensions.
    :return: DescriptiveResult with weight matrix.

    .. epigraph:: "Infinite Tsukuyomi!" -- Madara Uchiha, Naruto
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    Zc = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    k = min(n_dims, len(S))
    W = Vt[:k].T * S[:k]
    return DescriptiveResult(
        name="bb_weight_matrix",
        value=k,
        extra={"weights": W.tolist(), "singular_values": S[:k].tolist(), "n_dims": k},
    )


bbwt = bb_weight_matrix


def cheatsheet() -> str:
    return "bb_weight_matrix({}) -> Blackbox weight matrix estimation."
