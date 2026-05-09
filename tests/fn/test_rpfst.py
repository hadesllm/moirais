"""Tests for rpfst — time to first placement."""
import pandas as pd
from moirais.fn.rpfst import rplace_first

def test_rpfst_basic(otis_df):
    result = rplace_first(otis_df, date_col="start_date")
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.rpfst import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
