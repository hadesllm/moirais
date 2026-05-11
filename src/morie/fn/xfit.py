"""DML-style cross-fitting splits."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cross_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int | None = None,
) -> DescriptiveResult:
    """Generate DML-style cross-fitting (sample-splitting) fold indices.

    Returns train/test index pairs for each fold, suitable for
    Double/Debiased Machine Learning nuisance estimation.

    :param X: Feature matrix of shape (n, p).
    :param y: Target vector of length n.
    :param n_folds: Number of folds.
    :param seed: Random seed for reproducibility.
    :return: DescriptiveResult with fold indices in ``extra['folds']``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if n < n_folds:
        raise ValueError(f"n_folds ({n_folds}) > n_samples ({n})")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        test_idx = indices[current : current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size :]])
        folds.append({"train": train_idx, "test": test_idx})
        current += size

    return DescriptiveResult(
        name="cross_fit",
        value=n_folds,
        extra={"folds": folds, "n": n, "n_folds": n_folds},
    )


def cheatsheet() -> str:
    return "cross_fit(X, y, n_folds) -> DML cross-fitting fold indices"


xfit = cross_fit
