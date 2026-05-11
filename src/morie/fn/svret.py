"""Retrospective proximity model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svret(voter, incumbent=None, candidates=None):
    """Retrospective proximity model.

    Parameters
    ----------
    voter : array-like
        Voter position.
    incumbent : array-like, optional
        Incumbent position (past performance proxy).
    candidates : array-like, optional
        Candidate positions (n_candidates x n_dims).

    Returns
    -------
    DescriptiveResult
    """
    voter = np.asarray(voter, dtype=float)
    if incumbent is None:
        incumbent = np.zeros_like(voter)
    else:
        incumbent = np.asarray(incumbent, dtype=float)
    retro_dist = float(np.sum((voter - incumbent) ** 2))
    if candidates is not None:
        candidates = np.asarray(candidates, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        dists = np.array([float(np.sum((voter - c) ** 2)) for c in candidates])
        combined = np.exp(-0.5 * (dists + 0.5 * retro_dist))
        stat = float(np.max(combined))
    else:
        stat = float(np.exp(-0.5 * retro_dist))
    return DescriptiveResult(
        name="svret",
        value=stat,
        extra={"retro_dist": retro_dist},
    )


short = "svret"
alias = "svret"
quote = "The spice must flow. -- Paul Atreides"
svret = svret


def cheatsheet() -> str:
    return "svret({}) -> Retrospective proximity model."
