"""Simple random sample from a DataFrame. 'It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle'."""

from moirais.sampling import simple_random_sample as _fn

srs = _fn
simple_random_sample = _fn


def cheatsheet() -> str:
    return "srs() -> Simple random sample from a DataFrame. 'Do or do not, there "
