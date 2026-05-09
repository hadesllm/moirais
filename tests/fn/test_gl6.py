"""Tests for gl6 — Guttman Lambda 6."""
from moirais.fn.gl6 import gl6

def test_gl6_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = gl6(mapq_df[items])
    assert isinstance(result, float)


def test_cheatsheet():
    from moirais.fn.gl6 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
