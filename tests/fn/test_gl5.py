"""Tests for gl5 — Guttman Lambda 5."""
from moirais.fn.gl5 import gl5

def test_gl5_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = gl5(mapq_df[items])
    assert isinstance(result, float)


def test_cheatsheet():
    from moirais.fn.gl5 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
