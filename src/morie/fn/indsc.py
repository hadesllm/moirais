# morie.fn -- function file (hadesllm/morie)
"""INDSCAL individual differences MDS"""

import numpy as np

from .._spatial_voting import indscal as _indscal
from ._containers import DescriptiveResult


def indscal_mds(dissimilarities_list=None, *, dims=2, n=10):
    """INDSCAL individual differences multidimensional scaling.

    Returns
    -------
    DescriptiveResult
    """
    if dissimilarities_list is None:
        rng = np.random.default_rng(0)
        dissimilarities_list = []
        for _ in range(3):
            pts = rng.standard_normal((n, 2))
            D = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=-1))
            dissimilarities_list.append(D)
    result = _indscal(dissimilarities_list, n_dims=dims)
    return DescriptiveResult(
        name="INDSCAL",
        value=result.get("stress", 0.0),
        extra={"dims": dims, "n_subjects": len(dissimilarities_list), **result},
    )


short = "indsc"
alias = "indscal_mds"
quote = "The more you know, the more you realize you don't know. -- Aristotle"
indscal_mds = indscal_mds


def cheatsheet() -> str:
    return "indscal_mds({}) -> INDSCAL individual differences MDS"
