# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrap resampling for any scalar statistic. 'Truly wonderful the mind of a child is.'."""

from morie.sampling import bootstrap_sample as _fn

btsmp = _fn
bootstrap_sample = _fn


def cheatsheet() -> str:
    return "btsmp() -> Bootstrap resampling for any scalar statistic. 'Truly wonder"
