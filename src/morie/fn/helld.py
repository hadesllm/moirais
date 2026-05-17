# morie.fn -- function file (hadesllm/morie)
"""Hellinger distance."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"


def hellinger_dist(p, q, **kwargs) -> ESRes:
    r"""
    Compute Hellinger distance between two probability distributions.

    .. math::

        H(P, Q) = \\frac{1}{\\sqrt{2}} \\sqrt{\\sum_i
        (\\sqrt{p_i} - \\sqrt{q_i})^2}

    Bounded in [0, 1].

    :param p: array-like, probability distribution.
    :param q: array-like, probability distribution.
    :return: ESRes with Hellinger distance.

    References
    ----------
    Hellinger E (1909). Neue Begrundung der Theorie quadratischer
    Formen von unendlichvielen Veranderlichen. Journal fur die
    reine und angewandte Mathematik, 136, 210-271.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")
    p = p / p.sum()
    q = q / q.sum()
    h = float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))
    return ESRes(measure="hellinger_distance", estimate=h, extra={"squared": h**2, "n_categories": len(p)})


helld = hellinger_dist


def cheatsheet() -> str:
    return "hellinger_dist({}) -> Hellinger distance."
