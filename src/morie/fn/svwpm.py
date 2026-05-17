"""Weighted proximity model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svwpm(voter, candidates=None, weights=None):
    """Weighted proximity model.

    Parameters
    ----------
    voter : array-like
        Voter position.
    candidates : array-like, optional
        Candidate positions (n_candidates x n_dims).
    weights : array-like, optional
        Issue weights.

    Returns
    -------
    DescriptiveResult
    """
    voter = np.asarray(voter, dtype=float)
    if candidates is None:
        candidates = np.zeros((1, len(voter)))
    else:
        candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    if weights is None:
        weights = np.ones(len(voter)) / len(voter)
    else:
        weights = np.asarray(weights, dtype=float)
    dists = np.array([float(np.sum(weights * (voter - c) ** 2)) for c in candidates])
    utilities = np.exp(-0.5 * dists)
    stat = float(np.max(utilities))
    return DescriptiveResult(
        name="svwpm",
        value=stat,
        extra={"n_candidates": len(candidates)},
    )


short = "svwpm"
alias = "svwpm"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
svwpm = svwpm


def cheatsheet() -> str:
    return "svwpm({}) -> Weighted proximity model."
