"""Typical set size and probability."""

__all__ = ["typst"]

import numpy as np


def typst(
    pmf: np.ndarray,
    n: int,
    *,
    epsilon: float = 0.1,
) -> dict:
    r"""
    Compute typical set properties for an i.i.d. source.

    The typical set A_epsilon^(n) contains sequences x^n such that:

    .. math::

        2^{-n(H+\\epsilon)} \\le p(x^n) \\le 2^{-n(H-\\epsilon)}

    Parameters
    ----------
    pmf : np.ndarray
        Source PMF, shape (k,). Must sum to 1.
    n : int
        Block length.
    epsilon : float
        Typicality parameter, epsilon > 0.

    Returns
    -------
    dict
        'entropy' (H(X), bits),
        'typical_set_size_lower' (|(1 - eps) * 2^{nH}|),
        'typical_set_size_upper' (2^{n(H+eps)}),
        'typical_prob_lower' (1 - epsilon for large n),
        'log2_total_sequences' (n * log2(|alphabet|)),
        'n', 'epsilon'.

    References
    ----------
    Cover & Thomas (2006). Elements of Information Theory, Ch. 3 (AEP).
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim != 1 or not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must be a 1-D array summing to 1.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    eps_c = 1e-300
    h = -np.sum(pmf[pmf > eps_c] * np.log2(pmf[pmf > eps_c]))

    k = len(pmf)
    log2_total = n * np.log2(k)

    size_upper = 2.0 ** (n * (h + epsilon))
    size_lower = max((1.0 - epsilon) * 2.0 ** (n * h), 1.0) if n * h < 1000 else np.inf

    if n * (h + epsilon) > 1000:
        size_upper = np.inf
    if n * h > 1000:
        size_lower = np.inf

    return {
        "entropy": h,
        "typical_set_size_lower": size_lower,
        "typical_set_size_upper": size_upper,
        "typical_prob_lower": 1.0 - epsilon,
        "log2_total_sequences": log2_total,
        "n": n,
        "epsilon": epsilon,
    }
