# morie.fn — function file (hadesllm/morie)
"""EM algorithm IRT estimation"""

import numpy as np

from .._spatial_voting import em_irt as _em_irt
from ._containers import DescriptiveResult


def em_irt_estimate(votes=None, *, dims=1, n=10, m=5, max_iter=20):
    """EM algorithm for IRT ideal point estimation.

    Returns
    -------
    DescriptiveResult
    """
    if votes is None:
        rng = np.random.default_rng(0)
        votes = (rng.random((n, m)) > 0.5).astype(float)
    votes = np.asarray(votes, dtype=float)
    result = _em_irt(votes, n_dims=dims, max_iter=max_iter)
    return DescriptiveResult(
        name="EM IRT",
        value=result.get("loglik", 0.0),
        extra={"dims": dims, "max_iter": max_iter, **result},
    )


short = "emirt"
alias = "em_irt_estimate"
quote = "Waste no more time arguing what a good person should be. Be one. — Marcus Aurelius"
em_irt_estimate = em_irt_estimate


def cheatsheet() -> str:
    return "em_irt_estimate({}) -> EM algorithm IRT estimation"
