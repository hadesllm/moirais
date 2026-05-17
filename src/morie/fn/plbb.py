# morie.fn -- function file (hadesllm/morie)
"""Plot Blackbox results for a dimension pair."""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_blackbox_result(positions, dim1: int = 0, dim2: int = 1) -> DescriptiveResult:
    """Prepare scatter data for Blackbox 2-D plot.

    :param positions: n x d respondent position matrix.
    :param dim1: First dimension index.
    :param dim2: Second dimension index.
    :return: DescriptiveResult with x, y arrays.

    .. epigraph:: Luck is what happens when preparation meets opportunity. -- Seneca
    """
    import numpy as np

    X = np.asarray(positions, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    x = X[:, dim1].tolist() if dim1 < X.shape[1] else [0.0] * X.shape[0]
    y = X[:, dim2].tolist() if dim2 < X.shape[1] else [0.0] * X.shape[0]
    return DescriptiveResult(
        name="plot_blackbox_result",
        value=X.shape[0],
        extra={"x": x, "y": y, "dim1": dim1, "dim2": dim2, "n": X.shape[0]},
    )


plbb = plot_blackbox_result


def cheatsheet() -> str:
    return "plot_blackbox_result({}) -> Plot Blackbox results for a dimension pair."
