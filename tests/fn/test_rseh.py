"""Tests for rseh — SEM with CI."""
from morie.fn.rseh import rseh

def test_rseh_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = rseh(mapq_df[items])
    assert isinstance(result, dict)
    assert "sem" in result


def test_cheatsheet():
    from morie.fn.rseh import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
