# morie.fn -- function file (hadesllm/morie)
"""Polya urn scheme simulation. 'Always two there are.'"""

from __future__ import annotations

import numpy as np


def polya_urn(
    n_draws: int = 100,
    initial_colors: dict[str, int] | None = None,
    reinforcement: int = 1,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Simulate a Pólya urn process.

    An urn initially contains balls of different colors. At each step:
    1. Draw a ball uniformly at random
    2. Return it plus :math:`s` additional balls of the same color

    This induces a "rich-get-richer" dynamic. If initial composition is
    :math:`(n_1, \ldots, n_k)`, the limiting proportion follows
    a symmetric Dirichlet distribution.

    .. math::

        \frac{n_k^{(t)}}{N^{(t)}} \xrightarrow{a.s.} \theta_k, \quad
        (\theta_1, \ldots, \theta_k) \sim \text{Dir}(n_1 / s, \ldots, n_k / s)

    :param n_draws: Number of draws. Default 100.
    :type n_draws: int
    :param initial_colors: Dictionary mapping color names to initial counts.
                          If None, defaults to {'red': 1, 'black': 1}.
    :type initial_colors: dict[str, int] | None
    :param reinforcement: Number of balls added after each draw. Default 1.
    :type reinforcement: int
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'draws' (sequence of colors), 'counts' (final counts),
             'proportions', 'trace' (counts over time).
    :rtype: dict

    References
    ----------
    Pólya G. (1930). Sur quelques points de la théorie des probabilités.
    *Ann. Inst. H. Poincaré*, 1, 117-161.
    """
    if rng is None:
        rng = np.random.default_rng()

    if initial_colors is None:
        initial_colors = {"red": 1, "black": 1}

    if n_draws <= 0:
        raise ValueError(f"n_draws must be > 0, got {n_draws}")
    if reinforcement <= 0:
        raise ValueError(f"reinforcement must be > 0, got {reinforcement}")

    colors = list(initial_colors.keys())
    counts = {color: count for color, count in initial_colors.items()}
    draws = []
    trace = {color: [counts[color]] for color in colors}

    for _ in range(n_draws):
        # Compute probabilities
        total = sum(counts.values())
        probs = np.array([counts[color] / total for color in colors], dtype=float)

        # Draw a color
        drawn_color = rng.choice(colors, p=probs)
        draws.append(drawn_color)

        # Add reinforcement
        counts[drawn_color] += reinforcement

        # Record trace
        for color in colors:
            trace[color].append(counts[color])

    final_counts = counts
    final_total = sum(final_counts.values())
    proportions = {color: count / final_total for color, count in final_counts.items()}

    return {
        "draws": draws,
        "counts": final_counts,
        "proportions": proportions,
        "trace": trace,
        "n_draws": n_draws,
        "reinforcement": reinforcement,
    }


polya = polya_urn


def cheatsheet() -> str:
    return "polya_urn(n_draws=100, initial_colors={'red': 1, 'black': 1}) -> Pólya urn"
