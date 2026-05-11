"""Tests for diflr — logistic regression DIF."""
import numpy as np
import pandas as pd
from morie.fn.diflr import diflr

def test_diflr_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    group = (mapq_df["gender"] == "Male").astype(int).values
    result = diflr(mapq_df[items], group)
    assert result is not None


def test_cheatsheet():
    from morie.fn.diflr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
