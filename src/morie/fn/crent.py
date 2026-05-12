# morie.fn -- function file (hadesllm/morie)
"""Cross-entropy."""

import numpy as np

from ._containers import ESRes


def cross_entropy(p: np.ndarray, q: np.ndarray) -> ESRes:
    r"""
    Compute cross-entropy H(P, Q).

    .. math::

        H(P, Q) = -\\sum_i p_i \\log q_i

    :param p: (k,) true probability distribution.
    :param q: (k,) predicted probability distribution.
    :return: ESRes with cross-entropy (in nats).
    :raises ValueError: If distributions have different lengths.

    References
    ----------
    Shannon CE (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27, 379-423.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")
    p = p / p.sum()
    q = q / q.sum()
    q = np.clip(q, 1e-15, 1.0)
    ce = float(-np.sum(p * np.log(q)))
    return ESRes(measure="cross_entropy", estimate=ce, extra={"n_categories": len(p)})


crent = cross_entropy


def cheatsheet() -> str:
    return "cross_entropy({}) -> Cross-entropy."
