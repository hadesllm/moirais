# moirais.fn — function file (hadesllm/moirais)
"""Hamming distance between binary vectors."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I have spoken. -- Kuiil"


def hamming_distance_sets(a, b, **kwargs) -> DescriptiveResult:
    """
    Compute the Hamming distance between two binary vectors.

    .. math::

        d_H(a, b) = \\sum_{i=1}^{n} \\mathbb{1}[a_i \\neq b_i]

    The normalized Hamming distance :math:`d_H / n` is the proportion of
    positions at which the two vectors differ.

    :param a: First binary vector (array-like of 0/1 or bool).
    :param b: Second binary vector (same length as a).
    :return: DescriptiveResult with Hamming distance as value.
    :raises ValueError: If vectors have different lengths.

    References
    ----------
    Hamming, R. W. (1950). Error detecting and error correcting codes.
    *Bell System Technical Journal*, 29(2), 147-160.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}.")

    n = len(a)
    dist = int(np.sum(a != b))
    normalized = dist / n if n > 0 else 0.0

    return DescriptiveResult(
        name="hamming_distance_sets",
        value=float(dist),
        extra={
            "hamming_distance": dist,
            "normalized": float(normalized),
            "n": n,
            "agreement_fraction": 1.0 - normalized,
        },
    )


hmngd = hamming_distance_sets


def cheatsheet() -> str:
    return "hamming_distance_sets({}) -> Hamming distance between binary vectors."
