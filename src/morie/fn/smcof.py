"""SMACOF majorization scaling"""

import numpy as np

from .._spatial_voting import smacof as _smacof
from ._containers import DescriptiveResult


def smacof_scale(dissimilarities=None, *, dims=2, n=10, max_iter=300):
    """SMACOF majorization algorithm for MDS.

    Returns
    -------
    DescriptiveResult
    """
    if dissimilarities is None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((n, 2))
        dissimilarities = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=-1))
    dissimilarities = np.asarray(dissimilarities, dtype=float)
    result = _smacof(dissimilarities, n_dims=dims, max_iter=max_iter)
    return DescriptiveResult(
        name="SMACOF",
        value=result.get("stress", 0.0),
        extra={"dims": dims, "max_iter": max_iter, **result},
    )


short = "smcof"
alias = "smacof_scale"
quote = "Waste no more time arguing what a good person should be. Be one. — Marcus Aurelius"
smacof_scale = smacof_scale


def cheatsheet() -> str:
    return "smacof_scale({}) -> SMACOF majorization scaling"
