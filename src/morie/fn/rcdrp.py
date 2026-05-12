# morie.fn -- function file (hadesllm/morie)
"""Drop influential votes. 'Hellzone Grenade!' -- Piccolo, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def drop_influential_votes(votes, threshold=0.9):
    """Filter out votes where a single option exceeds threshold agreement.

    Parameters
    ----------
    votes : array-like
        Binary vote matrix (n_voters x n_items).
    threshold : float
        If proportion voting the same way exceeds this, drop the item.

    Returns
    -------
    DescriptiveResult
        value = filtered vote matrix, extra has dropped item indices.
    """
    import numpy as np

    V = np.asarray(votes, dtype=float)
    n = V.shape[0]
    keep = []
    dropped = []
    for j in range(V.shape[1]):
        prop = max(np.mean(V[:, j]), 1 - np.mean(V[:, j]))
        if prop <= threshold:
            keep.append(j)
        else:
            dropped.append(j)
    V_filtered = V[:, keep] if keep else V[:, :0]
    return DescriptiveResult(
        name="drop_influential_votes",
        value=V_filtered,
        extra={"n_kept": len(keep), "n_dropped": len(dropped), "dropped_indices": dropped},
    )


rcdrp = drop_influential_votes


def cheatsheet() -> str:
    return "drop_influential_votes({}) -> Drop influential votes. 'Hellzone Grenade!' -- Piccolo, Drag"
