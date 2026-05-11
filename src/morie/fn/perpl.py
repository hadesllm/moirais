# morie.fn — function file (hadesllm/morie)
"""Perplexity."""

import numpy as np

from ._containers import ESRes


def perplexity(log_probs, **kwargs) -> ESRes:
    """
    Compute perplexity from log-probabilities.

    .. math::

        PP = 2^{H} = 2^{-\\frac{1}{N}\\sum_i \\log_2 p(x_i)}

    :param log_probs: array-like of log-probabilities (natural log).
    :return: ESRes with perplexity.

    References
    ----------
    Jelinek F et al. (1977). Perplexity -- a measure of the difficulty
    of speech recognition tasks. JASA, 62(S1), S63.
    """
    lp = np.asarray(log_probs, dtype=np.float64).ravel()
    if len(lp) < 1:
        raise ValueError("Need at least 1 log-probability.")
    avg_nll = -float(np.mean(lp)) / np.log(2)
    pp = float(2.0 ** avg_nll)
    return ESRes(
        measure="perplexity",
        estimate=pp,
        n=len(lp),
        extra={"cross_entropy_bits": avg_nll},
    )


perpl = perplexity


def cheatsheet() -> str:
    return "perplexity(log_probs) -> Perplexity from log-probabilities."
