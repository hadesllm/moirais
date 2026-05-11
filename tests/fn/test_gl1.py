"""Tests for gl1 — Guttman Lambda 1."""
from morie.fn.gl1 import gl1

def test_gl1_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = gl1(mapq_df[items])
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_cheatsheet():
    from morie.fn.gl1 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
