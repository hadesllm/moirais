# morie.fn -- function file (hadesllm/morie)
"""Dimensionality test for spatial voting (Armstrong Ch 7)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dimensionality_test", "dimrd"]


def dimensionality_test(x, threshold: float = 1.0):
    """Cattell scree-criterion dimensionality test on an agreement /
    vote / configuration matrix.

    Build symmetric input S:
      - if x is (n, n) symmetric, S = x;
      - else S = corr(x) on (n, m) vote / score matrix.
    Decompose S = U Λ Uᵀ, then count eigenvalues with λ > threshold
    (Kaiser 1960 rule) as the number of latent dimensions.

    Returns
    -------
    RichResult with keys: n_dims, eigenvalues, threshold, scree_gap
    """
    M = np.asarray(x, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n, m = M.shape
    if n == m and np.allclose(M, M.T, atol=1e-9):
        S = (M + M.T) / 2
    else:
        if m < 2:
            return RichResult(payload={"n_dims": 0,
                                       "eigenvalues": np.array([]),
                                       "threshold": float(threshold),
                                       "scree_gap": np.nan,
                                       "method": "dimensionality_test"})
        # Use correlation matrix of items (columns); handle constant cols
        S = np.corrcoef(M, rowvar=False)
        if np.any(np.isnan(S)):
            S = np.nan_to_num(S, nan=0.0)
            np.fill_diagonal(S, 1.0)
    eigvals = np.linalg.eigvalsh((S + S.T) / 2)[::-1]
    n_dims = int(np.sum(eigvals > threshold))
    # Scree gap: largest drop between successive eigenvalues
    gaps = -np.diff(eigvals)
    scree_gap_idx = int(np.argmax(gaps)) + 1 if gaps.size else 0
    return RichResult(
        title="Dimensionality test (Kaiser scree)",
        summary_lines=[("n dimensions (λ > %.2f)" % threshold, n_dims),
                       ("Largest-gap k* (scree elbow)", scree_gap_idx),
                       ("Top eigenvalues",
                        list(np.round(eigvals[:min(5, eigvals.size)], 4)))],
        interpretation=(
            f"Kaiser rule suggests {n_dims} spatial dimension(s); scree "
            f"elbow at k* = {scree_gap_idx}."),
        payload={"n_dims": n_dims, "eigenvalues": eigvals,
                 "threshold": float(threshold),
                 "scree_gap": int(scree_gap_idx),
                 "method": "dimensionality_test"},
    )


dimrd = dimensionality_test


def cheatsheet():
    return "dimrd: Kaiser/scree dimensionality test on eigenvalues."


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> X = rng.normal(size=(50, 5))
# >>> r = dimensionality_test(X)
# >>> assert r["n_dims"] >= 1
