# moirais.fn — function file (hadesllm/moirais)
"""Matching Pursuit sparse signal decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def matching_pursuit_fn(x: np.ndarray, n_atoms: int = 10) -> DescriptiveResult:
    """Decompose signal using Matching Pursuit algorithm.

    :param x: 1-D input signal.
    :param n_atoms: Maximum number of atoms to use (default 10).
    :return: DescriptiveResult with n_atoms_used as value and decomposition in extra.
    """
    from moirais._decompose import matching_pursuit

    x = np.asarray(x, dtype=float).ravel()
    result = matching_pursuit(x, n_atoms=n_atoms)
    return DescriptiveResult(
        name="matching_pursuit",
        value=result.get("n_atoms_used", n_atoms),
        extra=result,
    )


mprst = matching_pursuit_fn


def cheatsheet() -> str:
    return "matching_pursuit_fn({}) -> Matching Pursuit sparse signal decomposition."
