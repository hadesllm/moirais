"""Asymmetric power index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def wvasym(weights_a, weights_b=None, quotas=None):
    """Asymmetric power index.

    Parameters
    ----------
    weights_a : array-like
        First set of weights.
    weights_b : array-like, optional
        Second set of weights.
    quotas : array-like, optional
        Quota thresholds.

    Returns
    -------
    DescriptiveResult
    """
    weights_a = np.asarray(weights_a, dtype=float)
    if weights_b is None:
        weights_b = np.ones_like(weights_a)
    else:
        weights_b = np.asarray(weights_b, dtype=float)
    if quotas is None:
        quotas = np.ones_like(weights_a)
    else:
        quotas = np.asarray(quotas, dtype=float)
    combined = weights_a * weights_b
    total = float(np.sum(combined))
    if total > 0:
        stat = float(np.sum(combined * quotas) / total)
    else:
        stat = 0.0
    return DescriptiveResult(
        name="wvasym",
        value=stat,
        extra={"total_weight": total},
    )


short = "wvasym"
alias = "wvasym"
quote = "The spice must flow. -- Paul Atreides"
wvasym = wvasym


def cheatsheet() -> str:
    return "wvasym({}) -> Asymmetric power index."
