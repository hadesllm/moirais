# moirais.fn — function file (hadesllm/moirais)
"""Delete-1 jackknife variance estimate. 'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius'."""

from moirais.sampling import jackknife_estimate as _fn

jksmp = _fn
jackknife_estimate = _fn


def cheatsheet() -> str:
    return "jksmp() -> Delete-1 jackknife variance estimate. 'Patience you must hav"
