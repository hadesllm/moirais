# morie.fn — function file (hadesllm/morie)
"""Patience is bitter, but its fruit is sweet. — Aristotle"""

from morie.sampling import pps_sample as _fn

ppssmp = _fn
pps_sample = _fn


def cheatsheet() -> str:
    return "Patience is bitter, but its fruit is sweet. — Aristotle"
