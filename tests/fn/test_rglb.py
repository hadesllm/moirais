"""Tests for rglb — greatest lower bound."""
from moirais.fn.rglb import rglb

def test_rglb_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = rglb(mapq_df[items])
    assert isinstance(result, float)


def test_cheatsheet():
    from moirais.fn.rglb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
