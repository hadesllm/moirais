"""Weighted directional model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svwdm(voter, candidates=None, weights=None):
    """Weighted directional model.

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
    dirs = np.array([float(np.sum(weights * voter * c)) for c in candidates])
    stat = float(np.max(dirs))
    return DescriptiveResult(
        name="svwdm",
        value=stat,
        extra={"n_candidates": len(candidates)},
    )


short = "svwdm"
alias = "svwdm"
quote = "The spice must flow. -- Paul Atreides"
svwdm = svwdm


def cheatsheet() -> str:
    return "svwdm({}) -> Weighted directional model."
