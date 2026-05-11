# morie.fn — function file (hadesllm/morie)
"""Kullback-Leibler divergence."""

import numpy as np

from ._containers import ESRes


def kl_divergence(p: np.ndarray, q: np.ndarray) -> ESRes:
    """
    Compute KL divergence D_KL(P || Q).

    .. math::

        D_{KL}(P \\| Q) = \\sum_i p_i \\log \\frac{p_i}{q_i}

    :param p: (k,) probability distribution P.
    :param q: (k,) probability distribution Q.
    :return: ESRes with KL divergence (in nats).
    :raises ValueError: If distributions have different lengths or Q has zeros where P is nonzero.

    References
    ----------
    Kullback S, Leibler RA (1951). On information and sufficiency.
    Annals of Mathematical Statistics, 22(1), 79-86.
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
    kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return ESRes(measure="kl_divergence", estimate=kl, extra={"n_categories": len(p)})


kldiv = kl_divergence


def cheatsheet() -> str:
    return "kl_divergence({}) -> Kullback-Leibler divergence."
