"""Tests for irtpc — partial credit model."""
import numpy as np
from morie.fn.irtpc import irtpc

def test_irtpc_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = irtpc(mapq_df[items].values)
    assert hasattr(result, "item_params")


def test_cheatsheet():
    from morie.fn.irtpc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
