"""Nested logit spatial model.

Category: Spatial
"""


import numpy as np

from ._containers import DescriptiveResult


def svnst(voter, candidates=None):
    """Nested logit spatial model.

    Parameters
    ----------
    voter : array-like
        Voter position.
    candidates : array-like, optional
        Candidate positions.

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
    dists = np.array([float(np.sum((voter - c) ** 2)) for c in candidates])
    utilities = np.exp(-0.5 * dists)
    stat = float(np.sum(utilities) / len(utilities))
    result = DescriptiveResult(
        name="svnst",
        value=stat,
        extra={"n_candidates": len(candidates)},
    )
    result.method = "nested_logit"
    return result


short = "svnst"
alias = "svnst"
quote = "The spice must flow. -- Paul Atreides"
svnst = svnst


def cheatsheet() -> str:
    return "svnst({}) -> Nested logit spatial model."
