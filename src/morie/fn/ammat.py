# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""A-M matrix setup: center and prepare for eigensolve."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_matrix_setup(Z) -> DescriptiveResult:
    """Center respondent x stimulus matrix for Aldrich-McKelvey eigensolve.

    :param Z: Respondent x stimulus perceptual data.
    :return: DescriptiveResult with centered matrix and means.

    .. epigraph:: A journey of a thousand miles begins with a single step. -- Lao Tzu
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    col_means = Z.mean(axis=0)
    row_means = Z.mean(axis=1)
    grand_mean = Z.mean()
    Zc = Z - col_means[None, :] - row_means[:, None] + grand_mean
    return DescriptiveResult(
        name="am_matrix_setup",
        value=float(grand_mean),
        extra={
            "centered": Zc.tolist(),
            "col_means": col_means.tolist(),
            "row_means": row_means.tolist(),
            "grand_mean": grand_mean,
            "shape": list(Z.shape),
        },
    )


ammat = am_matrix_setup


def cheatsheet() -> str:
    return "am_matrix_setup({}) -> A-M matrix setup: center and prepare for eigensolve."
