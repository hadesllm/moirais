"""Cross-entropy."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Numbers have life; they're not just symbols on paper. — Shakuntala Devi"


def cross_entropy(p, q, **kwargs) -> ESRes:
    """
    Compute cross-entropy H(P, Q).

    .. math::

        H(P, Q) = -\\sum_i p_i \\log q_i

    :param p: array-like, true distribution.
    :param q: array-like, predicted distribution.
    :return: ESRes with cross-entropy in nats.

    References
    ----------
    Shore JE, Johnson RW (1980). Axiomatic derivation of the
    principle of maximum entropy. IEEE Transactions on Information
    Theory, 26(1), 26-37.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    if np.any(q[mask] <= 0):
        raise ValueError("q must be > 0 wherever p > 0.")
    ce = -float(np.sum(p[mask] * np.log(q[mask])))
    return ESRes(
        measure="cross_entropy",
        estimate=ce,
        extra={"n_categories": len(p)},
    )


xent = cross_entropy


def cheatsheet() -> str:
    return "cross_entropy({}) -> Cross-entropy."
