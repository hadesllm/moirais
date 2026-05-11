"""Tests for altm — alert timeline."""
import pandas as pd
from morie.fn.altm import altmrng

def test_altm_basic(otis_df):
    result = altmrng(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.altm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
