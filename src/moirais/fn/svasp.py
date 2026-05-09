"""Asymmetric utility with salience weights.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svasp(position, ideal=None, saliences=None):
    """Asymmetric utility with salience weights.

    Parameters
    ----------
    position : array-like
        Current position in issue space.
    ideal : array-like, optional
        Ideal point.
    saliences : array-like, optional
        Salience weights per dimension.

    Returns
    -------
    DescriptiveResult
    """
    position = np.asarray(position, dtype=float)
    if ideal is None:
        ideal = np.zeros_like(position)
    else:
        ideal = np.asarray(ideal, dtype=float)
    if saliences is None:
        saliences = np.ones(len(position)) / len(position)
    else:
        saliences = np.asarray(saliences, dtype=float)
    dist_sq = float(np.sum(saliences * (position - ideal) ** 2))
    stat = float(np.exp(-0.5 * dist_sq))
    return DescriptiveResult(
        name="svasp",
        value=stat,
        extra={"dist_sq": dist_sq},
    )


short = "svasp"
alias = "svasp"
quote = "The spice must flow. -- Paul Atreides"
svasp = svasp


def cheatsheet() -> str:
    return "svasp({}) -> Asymmetric utility with salience weights."
