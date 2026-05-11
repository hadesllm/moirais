"""Tests for difgn — DIF by gender."""
from morie.fn.difgn import difgn

def test_difgn_basic(mapq_df):
    result = difgn(mapq_df, gender_col="gender")
    assert result is not None


def test_cheatsheet():
    from morie.fn.difgn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
