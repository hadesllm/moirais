# morie.fn — function file (hadesllm/morie)
"""Polya tree prior construction. 'Tree, I am not.'"""

from __future__ import annotations

import numpy as np


def polya_tree_construction(
    alpha_param: float = 1.0,
    max_level: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Construct a Pólya tree prior on :math:`[0, 1]`.

    A Pólya tree partitions :math:`[0, 1]` recursively into dyadic intervals,
    assigning probabilities via stick-breaking at each node.

    .. math::

        P(B_0) = V_\emptyset, \quad P(B_1 | B_0^c) = V_{\emptyset,R}, \quad \ldots

    where :math:`V \sim \text{Beta}(\alpha, \alpha)` at each node.

    :param alpha_param: Concentration (alpha > 0).
    :param max_level: Maximum tree depth.
    :param rng: Random number generator.
    :return: Dictionary with 'probabilities', 'intervals', 'tree_structure'.
    """
    if rng is None:
        rng = np.random.default_rng()

    if alpha_param <= 0:
        raise ValueError(f"alpha_param must be > 0, got {alpha_param}")

    tree = {}
    intervals = {}
    probabilities = {}

    def build_tree(node_id, left, right, level):
        if level >= max_level:
            return

        # Draw stick-breaking weight
        v = rng.beta(alpha_param, alpha_param)
        prob_left = v
        prob_right = 1.0 - v

        tree[f"{node_id}_L"] = prob_left
        tree[f"{node_id}_R"] = prob_right

        mid = (left + right) / 2
        intervals[f"{node_id}_L"] = (left, mid)
        intervals[f"{node_id}_R"] = (mid, right)

        probabilities[f"{node_id}_L"] = prob_left
        probabilities[f"{node_id}_R"] = prob_right

        # Recurse
        build_tree(f"{node_id}_L", left, mid, level + 1)
        build_tree(f"{node_id}_R", mid, right, level + 1)

    build_tree("root", 0.0, 1.0, 0)

    return {
        "tree": tree,
        "intervals": intervals,
        "probabilities": probabilities,
        "alpha": alpha_param,
        "max_level": max_level,
        "n_nodes": len(tree),
    }


polyt = polya_tree_construction


def cheatsheet() -> str:
    return "polya_tree_construction(alpha_param=1.0, max_level=10) -> Polya tree prior"
