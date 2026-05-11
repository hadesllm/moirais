"""Tests for rmdc — minimal detectable change."""
from morie.fn.rmdc import rmdc

def test_rmdc_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = rmdc(mapq_df[items])
    assert isinstance(result, float)
    assert result >= 0


def test_cheatsheet():
    from morie.fn.rmdc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
