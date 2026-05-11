"""Tests for rsem — standard error of measurement."""
from morie.fn.rsem import rsem

def test_rsem_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = rsem(mapq_df[items])
    assert isinstance(result, float)
    assert result >= 0


def test_cheatsheet():
    from morie.fn.rsem import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
