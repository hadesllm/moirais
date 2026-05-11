# morie.fn — function file (hadesllm/morie)
"""Dictionary Learning via K-SVD."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dctln_fn(X: np.ndarray, n_atoms: int = 10, n_iter: int = 50, sparsity: int = 3) -> DescriptiveResult:
    """Learn a dictionary from data via K-SVD algorithm.

    :param X: 2-D array (features x samples).
    :param n_atoms: Number of dictionary atoms (default 10).
    :param n_iter: Number of iterations (default 50).
    :param sparsity: Max non-zero coefficients per sample (default 3).
    :return: DescriptiveResult with atom count and dictionary/codes.
    """
    from morie._decompose import dictionary_learning

    X = np.asarray(X, dtype=float)
    D, codes = dictionary_learning(X, n_atoms=n_atoms, n_iter=n_iter, sparsity=sparsity)
    return DescriptiveResult(
        name="dictionary_learning",
        value=n_atoms,
        extra={"dictionary": D, "codes": codes},
    )


dctln = dctln_fn


def cheatsheet() -> str:
    return "dctln_fn({}) -> Dictionary Learning via K-SVD."
