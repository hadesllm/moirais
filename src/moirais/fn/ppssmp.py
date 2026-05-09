# moirais.fn — function file (hadesllm/moirais)
"""Patience is bitter, but its fruit is sweet. — Aristotle"""

from moirais.sampling import pps_sample as _fn

ppssmp = _fn
pps_sample = _fn


def cheatsheet() -> str:
    return "Patience is bitter, but its fruit is sweet. — Aristotle"
