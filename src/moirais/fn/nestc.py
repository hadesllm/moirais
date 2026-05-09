# moirais.fn — function file (hadesllm/moirais)
"""Nested cross-validation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    inner_k: int = 5,
    outer_k: int = 5,
    seed: int | None = None,
) -> DescriptiveResult:
    """Generate nested cross-validation fold indices.

    Outer folds for unbiased performance estimation, inner folds for
    hyperparameter tuning.

    :param X: Feature matrix of shape (n, p).
    :param y: Target vector of length n.
    :param inner_k: Number of inner folds.
    :param outer_k: Number of outer folds.
    :param seed: Random seed.
    :return: DescriptiveResult with nested fold structure in ``extra['folds']``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if n < outer_k:
        raise ValueError(f"outer_k ({outer_k}) > n_samples ({n})")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    outer_sizes = np.full(outer_k, n // outer_k, dtype=int)
    outer_sizes[: n % outer_k] += 1

    folds = []
    current = 0
    for o_size in outer_sizes:
        test_idx = indices[current : current + o_size]
        train_idx = np.concatenate([indices[:current], indices[current + o_size :]])
        current += o_size

        n_train = len(train_idx)
        inner_sizes = np.full(inner_k, n_train // inner_k, dtype=int)
        inner_sizes[: n_train % inner_k] += 1

        inner_folds = []
        i_current = 0
        for i_size in inner_sizes:
            i_test = train_idx[i_current : i_current + i_size]
            i_train = np.concatenate([train_idx[:i_current], train_idx[i_current + i_size :]])
            inner_folds.append({"train": i_train, "test": i_test})
            i_current += i_size

        folds.append({"outer_train": train_idx, "outer_test": test_idx, "inner_folds": inner_folds})

    return DescriptiveResult(
        name="nested_cv",
        value=outer_k,
        extra={"folds": folds, "n": n, "inner_k": inner_k, "outer_k": outer_k},
    )


def cheatsheet() -> str:
    return "nested_cv(X, y, inner_k, outer_k) -> nested CV fold indices"


nestc = nested_cv
