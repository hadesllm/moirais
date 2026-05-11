"""Tests for irt2p — 2PL model."""
import numpy as np
from morie.fn.irt2p import irt2p

def test_irt2p_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = irt2p(mapq_df[items].values)
    assert hasattr(result, "item_params")


def test_cheatsheet():
    from morie.fn.irt2p import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
