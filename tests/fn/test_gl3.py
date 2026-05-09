"""Tests for gl3 — Guttman Lambda 3 (= alpha)."""
from moirais.fn.gl3 import gl3

def test_gl3_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = gl3(mapq_df[items])
    assert isinstance(result, float)


def test_cheatsheet():
    from moirais.fn.gl3 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
