# morie.fn — function file (hadesllm/morie)
"""Post-stratification weights to match population distribution.

'The belonging you seek is not behind you, it is ahead.' -- Maz Kanata."""

from morie.survey import poststratification_weights as _fn

pstrat = _fn
poststratification_weights = _fn


def cheatsheet() -> str:
    return "pstrat() -> Post-stratification weights to match population distribution"
