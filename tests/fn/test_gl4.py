"""Tests for gl4 — Guttman Lambda 4."""
from morie.fn.gl4 import gl4

def test_gl4_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = gl4(mapq_df[items])
    assert isinstance(result, float)


def test_cheatsheet():
    from morie.fn.gl4 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
