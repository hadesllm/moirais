# morie.fn -- function file (hadesllm/morie)
"""Heatmap data for issue weight matrix."""

from __future__ import annotations

from ._containers import DescriptiveResult


def heatmap_issue_weights(weight_matrix) -> DescriptiveResult:
    """Flatten weight matrix for heatmap visualization.

    :param weight_matrix: Issue x dimension weight matrix.
    :return: DescriptiveResult with flattened data.

    .. epigraph:: "Kaizoku ou ni ore wa naru!" -- Monkey D. Luffy, One Piece
    """
    import numpy as np

    W = np.asarray(weight_matrix, dtype=float)
    return DescriptiveResult(
        name="heatmap_issue_weights",
        value=W.shape[0],
        extra={
            "weights": W.tolist(),
            "n_issues": W.shape[0],
            "n_dims": W.shape[1] if W.ndim > 1 else 1,
            "max_abs": float(np.max(np.abs(W))),
        },
    )


htmap = heatmap_issue_weights


def cheatsheet() -> str:
    return "heatmap_issue_weights({}) -> Heatmap data for issue weight matrix."
