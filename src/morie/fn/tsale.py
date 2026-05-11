"""Tsallis entropy (non-extensive)."""

__all__ = ["tsale"]

import numpy as np


def tsale(pmf: np.ndarray, q: float) -> dict:
    """
    Compute Tsallis entropy of order q.

    .. math::

        S_q(X) = \\frac{1}{q - 1}
        \\left(1 - \\sum_x p(x)^q \\right)

    For q -> 1, this converges to Shannon entropy (in nats, divided by ln 2).

    Parameters
    ----------
    pmf : np.ndarray
        Probability mass function, shape (n,). Must sum to 1.
    q : float
        Entropic index, q > 0.

    Returns
    -------
    dict
        'entropy' (float, Tsallis entropy in natural units),
        'entropy_bits' (float, converted to bits),
        'q', 'shannon_entropy' (bits, for comparison).

    Raises
    ------
    ValueError
        If pmf invalid or q <= 0.

    References
    ----------
    Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs
    statistics. J. Stat. Phys., 52(1-2), 479-487.
    """
    pmf = np.asarray(pmf, dtype=np.float64).ravel()
    if not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must sum to 1.")
    if np.any(pmf < 0):
        raise ValueError("pmf entries must be non-negative.")
    if q <= 0:
        raise ValueError("q must be > 0.")

    eps = 1e-300
    p_pos = pmf[pmf > eps]
    shannon = float(-np.sum(p_pos * np.log2(p_pos)))

    if np.isclose(q, 1.0):
        s_q = float(-np.sum(p_pos * np.log(p_pos)))
    else:
        s_q = float((1.0 - np.sum(p_pos ** q)) / (q - 1.0))

    s_q_bits = s_q / np.log(2) if not np.isclose(q, 1.0) else shannon

    return {
        "entropy": s_q,
        "entropy_bits": s_q_bits,
        "q": q,
        "shannon_entropy": shannon,
    }
