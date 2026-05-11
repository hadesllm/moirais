"""Tests for irt1p — 1PL Rasch model."""
import numpy as np
from morie.fn.irt1p import irt1p

def test_irt1p_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = irt1p(mapq_df[items].values)
    assert hasattr(result, "item_params")


def test_cheatsheet():
    from morie.fn.irt1p import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
