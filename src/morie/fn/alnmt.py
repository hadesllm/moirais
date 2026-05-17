# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""alpha-NOMINATE MCMC estimation"""

import numpy as np

from .._spatial_voting import alpha_nominate as _alpha_nominate
from ._containers import DescriptiveResult


def alpha_nominate_score(votes=None, *, dims=1, n=10, m=5, n_iter=20):
    """alpha-NOMINATE MCMC ideal point estimation.

    Returns
    -------
    DescriptiveResult
    """
    if votes is None:
        rng = np.random.default_rng(0)
        votes = (rng.random((n, m)) > 0.5).astype(float)
    votes = np.asarray(votes, dtype=float)
    result = _alpha_nominate(votes, n_dims=dims, n_samples=n_iter, burn_in=10)
    return DescriptiveResult(
        name="alpha-NOMINATE",
        value=result.get("alpha", 0.0),
        extra={"dims": dims, "n_iter": n_iter, **result},
    )


short = "alnmt"
alias = "alpha_nominate_score"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
alpha_nominate_score = alpha_nominate_score


def cheatsheet() -> str:
    return "alpha_nominate_score({}) -> alpha-NOMINATE MCMC estimation"
