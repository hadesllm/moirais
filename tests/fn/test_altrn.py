"""Tests for altrn — alert transition matrix."""
import pandas as pd
from moirais.fn.altrn import altrans

def test_altrn_basic(otis_df):
    result = altrans(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.altrn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
