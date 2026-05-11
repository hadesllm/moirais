# morie.fn — function file (hadesllm/morie)
"""Kish effective sample size from survey weights.

'In a dark place we find ourselves, and a little more knowledge lights our way.'."""

from morie.sampling import effective_sample_size as _fn

ess_s = _fn
effective_sample_size = _fn


def cheatsheet() -> str:
    return "ess_s() -> Kish effective sample size from survey weights. 'In a dark p"
