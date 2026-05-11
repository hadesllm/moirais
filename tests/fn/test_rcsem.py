"""Tests for rcsem — conditional SEM."""
import pandas as pd
from morie.fn.rcsem import rcsem

def test_rcsem_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = rcsem(mapq_df[items])
    assert isinstance(result, (pd.DataFrame, dict))


def test_cheatsheet():
    from morie.fn.rcsem import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
