# morie.fn -- function file (hadesllm/morie)
"""Jaccard similarity index."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Knowing others is intelligence; knowing yourself is true wisdom. -- Lao Tzu"


def jaccard_similarity(set_a, set_b, **kwargs) -> DescriptiveResult:
    r"""
    Compute the Jaccard similarity index between two sets.

    .. math::

        J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    Ranges from 0 (disjoint) to 1 (identical). Also computes the Jaccard
    distance :math:`d_J = 1 - J`.

    Works with sets, lists, or binary arrays.

    :param set_a: First set (iterable or binary array).
    :param set_b: Second set (iterable or binary array).
    :return: DescriptiveResult with Jaccard index as value.
    :raises ValueError: If both sets are empty.

    References
    ----------
    Jaccard, P. (1912). The distribution of the flora in the alpine zone.
    *New Phytologist*, 11(2), 37-50.
    """
    if isinstance(set_a, np.ndarray) and isinstance(set_b, np.ndarray):
        a = set_a.astype(bool).ravel()
        b = set_b.astype(bool).ravel()
        if len(a) != len(b):
            raise ValueError("Binary arrays must have the same length.")
        intersection = int(np.sum(a & b))
        union = int(np.sum(a | b))
    else:
        a = set(set_a)
        b = set(set_b)
        intersection = len(a & b)
        union = len(a | b)

    if union == 0:
        raise ValueError("Both sets are empty; Jaccard index is undefined.")

    j = intersection / union

    return DescriptiveResult(
        name="jaccard_similarity",
        value=float(j),
        extra={
            "jaccard_index": float(j),
            "jaccard_distance": 1.0 - float(j),
            "intersection_size": intersection,
            "union_size": union,
        },
    )


jaccr = jaccard_similarity


def cheatsheet() -> str:
    return "jaccard_similarity({}) -> Jaccard similarity index."
