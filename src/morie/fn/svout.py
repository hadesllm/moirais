"""Ordered categorical utility.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svout(position, ideal=None, thresholds=None):
    """Ordered categorical utility.

    Parameters
    ----------
    position : array-like
        Current position.
    ideal : array-like, optional
        Ideal point.
    thresholds : array-like, optional
        Category thresholds.

    Returns
    -------
    DescriptiveResult
    """
    position = np.asarray(position, dtype=float)
    if ideal is None:
        ideal = np.zeros_like(position)
    else:
        ideal = np.asarray(ideal, dtype=float)
    dist = float(np.sqrt(np.sum((position - ideal) ** 2)))
    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=float)
        category = int(np.searchsorted(thresholds, dist))
        stat = float(len(thresholds) - category) / float(len(thresholds))
    else:
        stat = float(np.exp(-0.5 * dist**2))
    return DescriptiveResult(
        name="svout",
        value=stat,
        extra={"dist": dist},
    )


short = "svout"
alias = "svout"
quote = "The only true wisdom is in knowing you know nothing. -- Socrates"
svout = svout


def cheatsheet() -> str:
    return "svout({}) -> Ordered categorical utility."
