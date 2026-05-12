# morie.fn — function file (hadesllm/morie)
"""Renyi entropy of order alpha."""

__all__ = ["renyh"]

import numpy as np
from ._richresult import RichResult


def renyh(pmf: np.ndarray, alpha: float) -> dict:
    r"""
    Compute Renyi entropy of order alpha.

    .. math::

        H_\\alpha(X) = \\frac{1}{1 - \\alpha}
        \\log_2 \\left( \\sum_x p(x)^\\alpha \\right)

    For alpha -> 1, this equals Shannon entropy. For alpha -> 0, log2 of
    support size. For alpha -> infinity, min-entropy.

    Parameters
    ----------
    pmf : np.ndarray
        Probability mass function, shape (n,). Must sum to 1.
    alpha : float
        Order parameter, alpha >= 0, alpha != 1 (alpha=1 returns Shannon).

    Returns
    -------
    dict
        'entropy' (float, bits), 'alpha', 'shannon_entropy' (for comparison).

    Raises
    ------
    ValueError
        If pmf invalid or alpha < 0.

    References
    ----------
    Renyi, A. (1961). On measures of entropy and information. Proc. 4th
    Berkeley Symp. Math. Stat. Prob., 1, 547-561.
    """
    pmf = np.asarray(pmf, dtype=np.float64).ravel()
    if not np.isclose(pmf.sum(), 1.0):
        raise ValueError("pmf must sum to 1.")
    if np.any(pmf < 0):
        raise ValueError("pmf entries must be non-negative.")
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")

    eps = 1e-300
    p_pos = pmf[pmf > eps]
    shannon = float(-np.sum(p_pos * np.log2(p_pos)))

    if np.isclose(alpha, 1.0):
        h_alpha = shannon
    elif np.isclose(alpha, 0.0):
        h_alpha = float(np.log2(np.sum(pmf > eps)))
    elif alpha == np.inf:
        h_alpha = float(-np.log2(np.max(pmf)))
    else:
        h_alpha = float(np.log2(np.sum(p_pos ** alpha)) / (1.0 - alpha))

    return RichResult(payload={"entropy": h_alpha, "alpha": alpha, "shannon_entropy": shannon})
