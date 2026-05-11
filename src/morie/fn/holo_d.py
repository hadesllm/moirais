# morie.fn — function file (hadesllm/morie)
"""DAG (Directed Acyclic Graph) diagram."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_dag(
    edges: Sequence[tuple[str, str]],
    *,
    ax: Any | None = None,
) -> Any:
    """
    Simple DAG diagram using matplotlib arrows.

    Nodes are arranged in a circular layout and directed edges are
    drawn as arrows between them.

    :param edges: Sequence of ``(source, target)`` string pairs.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_dag requires matplotlib. Install via: pip install matplotlib")
        return None

    nodes = list(dict.fromkeys(n for edge in edges for n in edge))
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / max(n, 1)
        pos[node] = (np.cos(angle), np.sin(angle))

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for node, (x, y) in pos.items():
        ax.annotate(
            node,
            (x, y),
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"),
        )

    for src, tgt in edges:
        sx, sy = pos[src]
        tx, ty = pos[tgt]
        dx, dy = tx - sx, ty - sy
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        shrink = 0.15
        ax.annotate(
            "",
            xy=(tx - shrink * dx / length, ty - shrink * dy / length),
            xytext=(sx + shrink * dx / length, sy + shrink * dy / length),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DAG")
    return fig


def cheatsheet() -> str:
    return "holo_dag({}) -> DAG (Directed Acyclic Graph) diagram."
