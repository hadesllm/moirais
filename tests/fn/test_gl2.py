"""Tests for gl2 — Guttman Lambda 2."""
from morie.fn.gl2 import gl2

def test_gl2_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = gl2(mapq_df[items])
    assert isinstance(result, float)


def test_cheatsheet():
    from morie.fn.gl2 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
