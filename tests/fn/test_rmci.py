"""Tests for rmci — reliable change index."""
import pandas as pd
import numpy as np
from morie.fn.rmci import rmci

def test_rmci_basic(rng):
    pre = rng.normal(50, 10, 30)
    post = pre + rng.normal(5, 3, 30)
    result = rmci(pre, post, sem=3.0)
    assert isinstance(result, (pd.DataFrame, dict))


def test_cheatsheet():
    from morie.fn.rmci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
