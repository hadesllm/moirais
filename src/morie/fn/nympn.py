# morie.fn — function file (hadesllm/morie)
"""Neyman-Pearson threshold."""

import numpy as np

from ._containers import ESRes

_QUOTE = "We suffer more often in imagination than in reality. — Seneca"


def neyman_pearson(scores, alpha: float = 0.05, **kwargs) -> ESRes:
    """
    Find the Neyman-Pearson decision threshold for a given false-alarm rate.

    Sorts *scores* and finds the value at the (1 - α) quantile, which
    controls the false-positive rate at level α.

    :param scores: array-like of test statistics or scores.
    :param alpha: Desired false-alarm rate (significance level).
    :return: ESRes with threshold value.

    References
    ----------
    Neyman J, Pearson ES (1933). On the problem of the most efficient
    tests of statistical hypotheses. Philosophical Transactions A,
    231, 289-337.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    threshold = float(np.quantile(scores, 1 - alpha))
    n_exceed = int(np.sum(scores > threshold))
    empirical_alpha = n_exceed / len(scores)
    return ESRes(
        measure="neyman_pearson_threshold",
        estimate=threshold,
        n=len(scores),
        extra={"alpha": alpha, "empirical_alpha": empirical_alpha, "n_exceed": n_exceed},
    )


nympn = neyman_pearson


def cheatsheet() -> str:
    return "neyman_pearson({}) -> Neyman-Pearson threshold."
