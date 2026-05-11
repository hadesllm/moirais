# morie.fn — function file (hadesllm/morie)
"""Profile a DataFrame: measurement levels, roles, summaries."""

from morie.dataset import profile_dataset as _fn

profds = _fn
profile_dataset = _fn


def cheatsheet() -> str:
    return "profds() -> Profile a DataFrame: measurement levels, roles, summaries."
