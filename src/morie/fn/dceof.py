# morie.fn — function file (hadesllm/morie)
"""Sorensen-Dice coefficient."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Numbers have life; they're not just symbols on paper. — Shakuntala Devi"


def dice_coefficient(set_a, set_b, **kwargs) -> DescriptiveResult:
    r"""
    Compute the Sorensen-Dice coefficient between two sets.

    .. math::

        DSC(A, B) = \\frac{2|A \\cap B|}{|A| + |B|}

    Ranges from 0 (disjoint) to 1 (identical). Closely related to the
    Jaccard index: :math:`DSC = 2J / (1 + J)`.

    Works with sets, lists, or binary arrays. Widely used in image
    segmentation evaluation and ecological community comparison.

    :param set_a: First set (iterable or binary array).
    :param set_b: Second set (iterable or binary array).
    :return: DescriptiveResult with Dice coefficient as value.
    :raises ValueError: If both sets are empty.

    References
    ----------
    Dice, L. R. (1945). Measures of the amount of ecologic association
    between species. *Ecology*, 26(3), 297-302.
    Sorensen, T. (1948). A method of establishing groups of equal amplitude
    in plant sociology. *Kongelige Danske Videnskabernes Selskab*, 5, 1-34.
    """
    if isinstance(set_a, np.ndarray) and isinstance(set_b, np.ndarray):
        a = set_a.astype(bool).ravel()
        b = set_b.astype(bool).ravel()
        if len(a) != len(b):
            raise ValueError("Binary arrays must have the same length.")
        intersection = int(np.sum(a & b))
        size_a = int(np.sum(a))
        size_b = int(np.sum(b))
    else:
        a = set(set_a)
        b = set(set_b)
        intersection = len(a & b)
        size_a = len(a)
        size_b = len(b)

    denom = size_a + size_b
    if denom == 0:
        raise ValueError("Both sets are empty; Dice coefficient is undefined.")

    dsc = 2.0 * intersection / denom

    return DescriptiveResult(
        name="dice_coefficient",
        value=float(dsc),
        extra={
            "dice_coefficient": float(dsc),
            "intersection_size": intersection,
            "size_a": size_a,
            "size_b": size_b,
        },
    )


dceof = dice_coefficient


def cheatsheet() -> str:
    return "dice_coefficient({}) -> Sorensen-Dice coefficient."
