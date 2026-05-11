"""Prospective proximity (expected utility).

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svpro(voter, candidates=None, uncertainty=None):
    """Prospective proximity (expected utility).

    Parameters
    ----------
    voter : array-like
        Voter position.
    candidates : array-like, optional
        Candidate positions (n_candidates x n_dims).
    uncertainty : array-like, optional
        Uncertainty per candidate.

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
    if uncertainty is None:
        uncertainty = np.ones(len(candidates)) * 0.1
    else:
        uncertainty = np.asarray(uncertainty, dtype=float)
    dists = np.array([float(np.sum((voter - c) ** 2)) for c in candidates])
    eu = np.exp(-0.5 * (dists + uncertainty**2))
    stat = float(np.max(eu))
    return DescriptiveResult(
        name="svpro",
        value=stat,
        extra={"n_candidates": len(candidates)},
    )


short = "svpro"
alias = "svpro"
quote = "The spice must flow. -- Paul Atreides"
svpro = svpro


def cheatsheet() -> str:
    return "svpro({}) -> Prospective proximity (expected utility)."
