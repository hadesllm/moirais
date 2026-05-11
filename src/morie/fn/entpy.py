# morie.fn — function file (hadesllm/morie)
"""Shannon entropy."""

import numpy as np

from ._containers import ESRes


def entropy(data: np.ndarray, base: float = 2.0) -> ESRes:
    """
    Compute Shannon entropy.

    .. math::

        H = -\\sum_i p_i \\log_b(p_i)

    If input sums to 1, treated as probabilities; otherwise treated
    as counts and normalised.

    :param data: Probabilities or counts.
    :param base: Logarithm base (2 = bits, e = nats).
    :return: ESRes with entropy value.
    :raises ValueError: If any value is negative.

    References
    ----------
    Shannon CE (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27, 379-423.
    """
    p = np.asarray(data, dtype=np.float64).ravel()
    if np.any(p < 0):
        raise ValueError("All values must be non-negative.")
    total = p.sum()
    if total <= 0:
        return ESRes(measure="entropy", estimate=0.0)
    p = p / total
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p) / np.log(base)))
    max_H = float(np.log(len(p)) / np.log(base)) if len(p) > 1 else 0.0
    return ESRes(
        measure="entropy",
        estimate=H,
        extra={"base": base, "max_entropy": max_H, "normalised": H / max_H if max_H > 0 else 0.0},
    )


entpy = entropy


def cheatsheet() -> str:
    return "entropy({}) -> Shannon entropy."
