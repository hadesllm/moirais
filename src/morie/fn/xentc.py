"""Cross-entropy H(P,Q)."""

import numpy as np

from ._containers import ESRes


def cross_entropy(p, q, **kwargs) -> ESRes:
    """
    Compute cross-entropy H(P, Q) = -sum p_i log q_i.

    .. math::

        H(P,Q) = -\\sum_i p_i \\log q_i

    :param p: array-like, true distribution.
    :param q: array-like, predicted distribution.
    :return: ESRes with cross-entropy in nats.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    if np.any(q[mask] <= 0):
        raise ValueError("Q must be > 0 wherever P > 0.")
    ce = -float(np.sum(p[mask] * np.log(q[mask])))
    return ESRes(
        measure="cross_entropy",
        estimate=ce,
        extra={"n_categories": len(p)},
    )


xentc = cross_entropy


def cheatsheet() -> str:
    return "cross_entropy(p, q) -> Cross-entropy H(P,Q)."
