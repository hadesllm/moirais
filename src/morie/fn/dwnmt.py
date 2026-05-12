# morie.fn -- function file (hadesllm/morie)
"""DW-NOMINATE ideal point estimation"""

import numpy as np

from .._spatial_voting import dw_nominate as _dw_nominate
from ._containers import DescriptiveResult


def dw_nominate_score(votes=None, *, dims=2, n=10, m=5):
    """DW-NOMINATE ideal point estimation from roll call votes.

    Returns
    -------
    DescriptiveResult
    """
    if votes is None:
        rng = np.random.default_rng(0)
        votes = (rng.random((n, m)) > 0.5).astype(float)
    votes = np.asarray(votes, dtype=float)
    result = _dw_nominate(votes, n_dims=dims)
    return DescriptiveResult(
        name="DW-NOMINATE",
        value=result.get("loglik", 0.0),
        extra={"dims": dims, "n_legislators": votes.shape[0], "n_votes": votes.shape[1], **result},
    )


short = "dwnmt"
alias = "dw_nominate_score"
quote = "Errors using inadequate data are much less than those using no data at all. -- Charles Babbage"
dw_nominate_score = dw_nominate_score


def cheatsheet() -> str:
    return "dw_nominate_score({}) -> DW-NOMINATE ideal point estimation"
