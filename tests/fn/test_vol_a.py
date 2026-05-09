"""Tests for vol_a — volatility by age."""
import pandas as pd
from moirais.fn.vol_a import vol_age

def test_vol_a_basic(otis_df):
    result = vol_age(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.vol_a import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
