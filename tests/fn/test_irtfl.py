"""Tests for irtfl — item fit statistics."""
import numpy as np
import pandas as pd
from moirais.fn.irtfl import irtfl

def test_irtfl_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE","EA","UA","ER")) and c[-1].isdigit()]
    data = mapq_df[items].values
    # Simple 1PL params: all difficulty = 0
    params = {items[i]: {"b": 0.0} for i in range(len(items))}
    result = irtfl(data, params)
    assert isinstance(result, (pd.DataFrame, dict))


def test_cheatsheet():
    from moirais.fn.irtfl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
