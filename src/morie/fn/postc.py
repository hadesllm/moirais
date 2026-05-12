# morie.fn -- function file (hadesllm/morie)
"""Posterior consistency rate under nonparametric model. 'Converge, the posterior must.'"""

from __future__ import annotations

import numpy as np


def posterior_consistency_rate(
    n: int,
    dimension: int = 1,
    smoothness: float = 2.0,
    metric: str = 'hellinger',
) -> float:
    r"""
    Compute minimax posterior consistency rate for nonparametric models.

    Based on Ghosal & van der Vaart theory. For Holder class with smoothness :math:`\beta`:

    .. math::

        \epsilon_n \sim n^{-\beta / (2\beta + d)}

    where d is the dimension.

    :param n: Sample size.
    :param dimension: Problem dimension (default 1).
    :param smoothness: Holder smoothness parameter :math:`\beta` (default 2.0).
    :param metric: Distance metric ('hellinger', 'l2', 'linf').
    :return: Consistency rate :math:`\epsilon_n`.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, got {dimension}")
    if smoothness <= 0:
        raise ValueError(f"smoothness must be > 0, got {smoothness}")

    # Standard minimax rate for Holder class
    exponent = -smoothness / (2 * smoothness + dimension)

    # Metric-specific constants
    if metric == 'hellinger':
        const = 1.0
    elif metric == 'l2':
        const = 1.0  # Similar to Hellinger for densities
    elif metric == 'linf':
        const = 1.2  # Slightly worse for sup norm
    else:
        const = 1.0

    rate = const * (n ** exponent)

    return float(rate)


def posterior_contraction_analysis(
    n: int,
    prior_type: str = 'dp',
    alpha: float = 1.0,
    dimension: int = 1,
) -> dict:
    r"""
    Analyze posterior contraction for common nonparametric priors.

    Returns contraction rate and supporting information.

    :param n: Sample size.
    :param prior_type: 'dp' (Dirichlet process), 'polya_tree', 'histogram'.
    :param alpha: Prior concentration.
    :param dimension: Problem dimension.
    :return: Dictionary with contraction rate and theory details.
    """
    if prior_type == 'dp':
        # DP mixture: rate n^{-1/(2+d)}
        rate = (n ** (-1.0 / (2.0 + dimension))) / np.sqrt(np.log(n))
        notes = "DP mixture (L2 metric, smooth density)"
    elif prior_type == 'polya_tree':
        # Polya tree: adaptive to smoothness
        rate = (n ** (-2.0 / (4.0 + dimension))) / np.log(n)
        notes = "Polya tree (adaptive to smoothness)"
    elif prior_type == 'histogram':
        # Histogram: slower rate
        rate = (n ** (-1.0 / (1.0 + dimension)))
        notes = "Histogram (naive binning)"
    else:
        rate = np.nan
        notes = "Unknown prior"

    return {
        "contraction_rate": float(rate),
        "prior_type": prior_type,
        "dimension": dimension,
        "alpha": alpha,
        "n": n,
        "notes": notes,
    }


postc = posterior_consistency_rate


def cheatsheet() -> str:
    return "posterior_consistency_rate(n, dimension=1, smoothness=2.0) -> consistency rate"
