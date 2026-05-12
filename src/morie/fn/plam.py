# morie.fn -- function file (hadesllm/morie)
"""Plot summary data for Aldrich-McKelvey results."""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_am_summary(positions, weights=None, bins: int = 30) -> DescriptiveResult:
    """Prepare histogram and density data for A-M position summary.

    :param positions: Estimated ideal points.
    :param weights: Optional respondent weights for density split.
    :param bins: Number of histogram bins.
    :return: DescriptiveResult with histogram data.

    .. epigraph:: "Beyond the horizon." -- Gol D. Roger, One Piece
    """
    import numpy as np

    pos = np.asarray(positions, dtype=float).ravel()
    counts, edges = np.histogram(pos[~np.isnan(pos)], bins=bins)
    result = {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "mean": float(np.nanmean(pos)),
        "std": float(np.nanstd(pos)),
    }
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        result["n_positive"] = int(np.sum(w > 0))
        result["n_negative"] = int(np.sum(w < 0))
    return DescriptiveResult(name="plot_am_summary", value=float(np.nanmean(pos)), extra=result)


plam = plot_am_summary


def cheatsheet() -> str:
    return "plot_am_summary({}) -> Plot summary data for Aldrich-McKelvey results."
