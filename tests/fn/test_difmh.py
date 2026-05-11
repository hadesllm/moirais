"""Tests for difmh — Mantel-Haenszel DIF."""
import pandas as pd
from morie.fn.difmh import difmh

def test_difmh_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = difmh(mapq_df[items], mapq_df["gender"])
    assert result is not None


def test_cheatsheet():
    from morie.fn.difmh import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
