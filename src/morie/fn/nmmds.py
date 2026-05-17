# morie.fn -- function file (hadesllm/morie)
"""Nonmetric multidimensional scaling"""

import numpy as np

from .._spatial_voting import nonmetric_mds as _nonmetric_mds
from ._containers import DescriptiveResult


def nonmetric_mds(dissimilarities=None, *, dims=2, n=10):
    """Nonmetric multidimensional scaling with ordinal regression.

    Returns
    -------
    DescriptiveResult
    """
    if dissimilarities is None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((n, 2))
        dissimilarities = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=-1))
    dissimilarities = np.asarray(dissimilarities, dtype=float)
    result = _nonmetric_mds(dissimilarities, n_dims=dims)
    return DescriptiveResult(
        name="Nonmetric MDS",
        value=result.get("stress", 0.0),
        extra={"dims": dims, **result},
    )


short = "nmmds"
alias = "nonmetric_mds"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
nonmetric_mds = nonmetric_mds


def cheatsheet() -> str:
    return "nonmetric_mds({}) -> Nonmetric multidimensional scaling"
