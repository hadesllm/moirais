# morie.fn — function file (hadesllm/morie)
"""Design effect (DEFF) from survey weights. 'The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert'."""

from morie.sampling import design_effect as _fn

deff = _fn
design_effect = _fn


def cheatsheet() -> str:
    return "deff() -> Design effect (DEFF) from survey weights. 'The dark side clo"
