"""Tests for irt3p — 3PL model."""
import numpy as np
from morie.fn.irt3p import irt3p

def test_irt3p_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    result = irt3p(mapq_df[items].values)
    assert hasattr(result, "item_params")


def test_cheatsheet():
    from morie.fn.irt3p import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
