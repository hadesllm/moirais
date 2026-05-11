"""Tests for alco — alert co-occurrence."""
import pandas as pd
from morie.fn.alco import alcooc

def test_alco_basic(otis_df):
    result = alcooc(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.alco import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
