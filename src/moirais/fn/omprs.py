# moirais.fn — function file (hadesllm/moirais)
"""Orthogonal Matching Pursuit sparse signal decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def omp_fn(x: np.ndarray, n_atoms: int = 10) -> DescriptiveResult:
    """Decompose signal using Orthogonal Matching Pursuit algorithm.

    :param x: 1-D input signal.
    :param n_atoms: Maximum number of atoms to use (default 10).
    :return: DescriptiveResult with n_atoms_used as value and decomposition in extra.
    """
    from moirais._decompose import orthogonal_matching_pursuit

    x = np.asarray(x, dtype=float).ravel()
    result = orthogonal_matching_pursuit(x, n_atoms=n_atoms)
    return DescriptiveResult(
        name="orthogonal_matching_pursuit",
        value=result.get("n_atoms_used", n_atoms),
        extra=result,
    )


omprs = omp_fn


def cheatsheet() -> str:
    return "omp_fn({}) -> Orthogonal Matching Pursuit sparse signal decomposition."
