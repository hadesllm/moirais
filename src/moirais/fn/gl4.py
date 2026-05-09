# moirais.fn — function file (hadesllm/moirais)
"""Guttman's Lambda 4 (maximum split-half reliability)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gl4(
    data: pd.DataFrame | np.ndarray,
    *,
    n_splits: int = 500,
    seed: int = 42,
) -> float:
    """Guttman's Lambda 4 — maximum split-half reliability.

    The maximum reliability obtainable from splitting the test into two
    halves.  For k <= 20, evaluates a random sample of splits; for
    larger k, uses a principal-component-guided split as approximation.

    For each split into halves A and B:
        r_split = 4 * cov(A, B) / var_total

    Lambda 4 = max(r_split) over all evaluated splits.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).
    n_splits : int
        Number of random splits to evaluate (default 500).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    float
        Lambda 4 coefficient.

    References
    ----------
    Guttman, L. (1945). A basis for analyzing test-retest reliability.
    *Psychometrika*, 10(4), 255-282.
    """
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2:
        return float("nan")

    total = X.sum(axis=1)
    total_var = np.var(total, ddof=1)

    if total_var < 1e-15:
        return float("nan")

    rng = np.random.RandomState(seed)
    best = -np.inf

    # Principal component split: assign items to halves by sign of
    # first eigenvector — this is usually near-optimal.
    C = np.cov(X, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(C)
    pc1 = eigvecs[:, -1]  # largest eigenvalue's eigenvector
    split_pc = pc1 >= 0
    # Ensure both halves have at least one item
    if split_pc.sum() == 0 or split_pc.sum() == k:
        split_pc[: k // 2] = True
        split_pc[k // 2 :] = False

    a_sum = X[:, split_pc].sum(axis=1)
    b_sum = X[:, ~split_pc].sum(axis=1)
    cov_ab = np.cov(a_sum, b_sum, ddof=1)[0, 1]
    best = max(best, 4.0 * cov_ab / total_var)

    # Random splits
    half = k // 2
    for _ in range(n_splits):
        perm = rng.permutation(k)
        idx_a = perm[:half]
        idx_b = perm[half:]
        a_sum = X[:, idx_a].sum(axis=1)
        b_sum = X[:, idx_b].sum(axis=1)
        cov_ab = np.cov(a_sum, b_sum, ddof=1)[0, 1]
        r_split = 4.0 * cov_ab / total_var
        if r_split > best:
            best = r_split

    return float(min(best, 1.0))


short = gl4


def cheatsheet() -> str:
    return "gl4({}) -> Guttman's Lambda 4 (maximum split-half reliability)."
