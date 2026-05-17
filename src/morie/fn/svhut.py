"""Hierarchical utility with nested issues.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svhut(positions, weights=None, issue_weights=None):
    """Hierarchical utility with nested issues.

    Parameters
    ----------
    positions : array-like
        Issue positions.
    weights : array-like, optional
        Weights per position.
    issue_weights : array-like, optional
        Issue-level weights.

    Returns
    -------
    DescriptiveResult
    """
    positions = np.asarray(positions, dtype=float)
    if weights is None:
        weights = np.ones_like(positions) / len(positions)
    else:
        weights = np.asarray(weights, dtype=float)
    if issue_weights is None:
        issue_weights = np.ones_like(positions) / len(positions)
    else:
        issue_weights = np.asarray(issue_weights, dtype=float)
    stat = float(np.sum(weights * issue_weights * positions))
    return DescriptiveResult(
        name="svhut",
        value=stat,
        extra={"n": len(positions)},
    )


short = "svhut"
alias = "svhut"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
svhut = svhut


def cheatsheet() -> str:
    return "svhut({}) -> Hierarchical utility with nested issues."
